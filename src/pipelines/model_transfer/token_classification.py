"""This modules contain labeler of words for model transfer pipeline"""

import json
import logging
from typing import Any, Iterable

from tqdm import tqdm
from transformers import AutoConfig, AutoModelForTokenClassification

from src.pipelines.transforms_base import PipelineTransformBase, batched, extract_key
from src.utils.model_context import use_hf_pipeline

logger = logging.getLogger(__file__)


class TokenClassificationTransform(PipelineTransformBase):
    """This class simply apply specified model to the given words
    and outputs labels for every word in the input. The
    main purpose of this class is to be used as an actual labeling
    step in the model transfer pipeline"""

    def __init__(
        self,
        model_path: str,
        batch_size: int = 32,
        device: int = 0,
        column_key: str = "tokens",
        label2id_save_path: str | None = None,
    ) -> None:
        super().__init__()

        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.column_key = column_key

        if label2id_save_path:
            config = AutoConfig.from_pretrained(self.model_path)
            with open(label2id_save_path, "w") as out:
                json.dump({"label2id": config.label2id}, out)

    def __call__(self, input: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        with use_hf_pipeline(
            "tokenwise-classification",
            self.model_path,
            pipeline_kwargs={
                "device": self.device,
                "batch_size": self.batch_size,
            },
            AutoModelClass=AutoModelForTokenClassification,
        ) as pipe:
            for batch in tqdm(batched(input, self.batch_size)):
                tokens = extract_key(batch, self.column_key)
                ner_out_iter = pipe(tokens)

                for ner_out, row in zip(ner_out_iter, batch):
                    labels = ["O"] * len(row[self.column_key])
                    try:
                        for ent in ner_out:
                            label = ent["entity_group"]
                            s = ent["start"]
                            e = ent["end"]

                            labels[s] = "B-" + label
                            for idx in range(s + 1, e):
                                labels[idx] = "I-" + label
                    except Exception:
                        logger.warn(
                            f"Fail to label tokens for the input {row}. Reason: {e}"
                        )

                    row["labels"] = labels
                    yield row
