"""This modules contains pipeline transforms which performs
NER labeling of an input sentence and output slotted sentence with
corresponding labels for every slot index. Also it contains a
pipeline transform for unslotting of the slotted sentence which outputs
entities"""

import json
import logging
import string
from typing import Any, Iterable

from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
)

from src.models.ner.tokenwise_pipeline import register_pipeline
from src.pipelines.transforms_base import PipelineTransformBase, batched, extract_key
from src.utils.model_context import use_hf_pipeline

logger = logging.getLogger(__file__)


class NERTransform(PipelineTransformBase):
    def __init__(
        self,
        model_path: str,
        device: int,
        batch_size: int,
        column_key: str,
        wordwise: bool = False,
        agg_straregy: str = "first",
        filter_punctuation: bool = False,
        symbols_to_filter_out: list[str] = list(string.punctuation),
        preserve_initials_on_filtering: bool = True,
        label2id_save_path: str | None = None,
        class_mapping: dict[str, str] | None = None,
    ) -> None:
        super().__init__()

        if filter_punctuation and wordwise:
            raise ValueError(
                """Filtering of punctuation on wordwise predicitions can
                corrupt some words. Please choose either wordwise or
                filter_punctuation"""
            )

        self.column_key = column_key
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.agg_straregy = agg_straregy
        self.label2id_path = label2id_save_path

        self.wordwise = wordwise
        if wordwise:
            register_pipeline()

        self.filter_punc = filter_punctuation
        self.preserve_initials = preserve_initials_on_filtering
        self.punctutation = set(symbols_to_filter_out)

        self.class_mapping = class_mapping
        if label2id_save_path and not class_mapping:
            config = AutoConfig.from_pretrained(self.model_path)
            with open(label2id_save_path, "w") as out:
                json.dump({"label2id": config.label2id}, out)

    @staticmethod
    def filter_punctuation_from_ner_out(
        ner_out: list[dict[str, Any]],
        symbols: set[str] = {",", ".", "!", "?"},
        preserve_initials: bool = True,
    ) -> list[dict[str, Any]]:
        """When NER model tokenizer doesn't support
            real words it splits them with use of heuristic and it leads to merged
            punctuation symbols and previous words which can cause a problem,
            because punctuation sign can/will be marked as an entity.
            This func filters out all punctuations from entities by removing
            specified symbols from the end of the entity.

        Args:
            ner_out (list[dict]): output of the NER model (has to contain start
            and end indices <=> fast tokenizer has to be used and aggregation
            straregy has to be one of the following: first, average or max).
            ner_out has to be sorted by start index (it is so by default).

            symbols: set[str]: symbols to be removed from the end of every entity

        Returns:
            list[dict]: modified NER model outputs where ending punctuation sign
            doesn't belong to any entity
        """

        result_ner_out = []

        check_initials = "." in symbols and preserve_initials

        for out_dict in ner_out:
            start_idx, end_idx = out_dict["start"], out_dict["end"]
            entity_len = end_idx - start_idx

            if entity_len > 1:
                last_symbol = out_dict["word"][-1]
                if last_symbol in symbols:
                    if check_initials and entity_len == 2 and last_symbol == ".":
                        result_ner_out.append(out_dict)
                    else:
                        result_ner_out.append(
                            {
                                "entity_group": out_dict["entity_group"],
                                "score": out_dict["score"],
                                "word": out_dict["word"][:-1],
                                "start": start_idx,
                                "end": end_idx - 1,
                            }
                        )
                else:
                    result_ner_out.append(out_dict)

        return result_ner_out

    @staticmethod
    def map_ner_out_to_entity(ner_out: dict[str, Any]) -> dict[str, Any]:
        return {
            "start_idx": ner_out["start"],
            "end_idx": ner_out["end"],
            "label": ner_out["entity_group"],
        }

    def map_labels(
        self, entities: Iterable[dict[str, Any]]
    ) -> Iterable[dict[str, Any]]:
        for entity in entities:
            label = entity["label"]
            if label in self.class_mapping:
                new_label = self.class_mapping[label]
                if new_label != "O":
                    entity["label"] = new_label
                else:  # skip entity
                    continue
            yield entity

    def __call__(self, input: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        with use_hf_pipeline(
            "tokenwise-classification" if self.wordwise else "ner",
            self.model_path,
            pipeline_kwargs={
                "device": self.device,
                "batch_size": self.batch_size,
                "aggregation_strategy": self.agg_straregy,
            },
            AutoModelClass=AutoModelForTokenClassification,
        ) as pipe:
            for batch in tqdm(batched(input, self.batch_size)):
                sentences = extract_key(batch, self.column_key)
                ner_out_iter = pipe(sentences)

                for ner_out, in_row in zip(ner_out_iter, batch):
                    if self.filter_punc:
                        ner_out = self.filter_punctuation_from_ner_out(
                            ner_out,
                            symbols=self.punctutation,
                            preserve_initials=self.preserve_initials,
                        )

                    entities = map(self.map_ner_out_to_entity, ner_out)
                    if self.class_mapping is not None:
                        entities = self.map_labels(entities)

                    out_key = "entities" if self.wordwise else "ner_out"
                    in_row[out_key] = list(entities)

                    yield in_row
