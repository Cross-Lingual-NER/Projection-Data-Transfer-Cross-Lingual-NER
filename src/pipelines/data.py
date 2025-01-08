"""This module contains pipeline transfroms which related to data
reading and writing in particular for arrow and HF dataset formats"""

from collections import defaultdict
import json
import os
import uuid
from glob import glob
from typing import Any, Iterable

import datasets
import pyarrow as pa

from src.pipelines.transforms_base import (
    PipelineTransformBase,
    batched,
    flatten_batch_dict,
)
from src.utils.iterators import primed


class LoadHFDataset(PipelineTransformBase):
    is_materialized = True

    def __init__(
        self,
        cfg_name: str | None = None,
        split: str | None = None,
        streaming: bool = False,
        label2id_save_path: str | None = None,
    ) -> None:
        self.cfg_name = cfg_name
        self.split = split
        self.streaming = streaming
        self.label2id_save_path = label2id_save_path

    def _save_labels2id_map(self, ds: datasets.Dataset) -> None:
        label2id = {
            label: id for id, label in enumerate(ds.features["ner_tags"].feature.names)
        }
        with open(self.label2id_save_path, "w") as out:
            json.dump({"label2id": label2id}, out)

    def __call__(self, ds_path: str) -> datasets.Dataset:
        if self.split == "MERGE_ALL":
            asked_split = None
        else:
            asked_split = self.split

        ds = datasets.load_dataset(
            ds_path,
            name=self.cfg_name,
            split=asked_split,
            streaming=self.streaming,
        )

        if self.split == "MERGE_ALL" and isinstance(ds, datasets.DatasetDict):
            ds = datasets.concatenate_datasets([ds[key] for key in ds.keys()])

        if self.label2id_save_path:
            self._save_labels2id_map(ds)

        return ds


class OpenArrow(PipelineTransformBase):
    is_materialized = True

    def __init__(self, split: str | None = None, concat_subsets: bool = False) -> None:
        """
        Args:
            split (str | None, optional): Optional name of dataset's split to use.
                Defaults to None.
            concat_subsets (bool, optional): wheteher concat all datasets splits into
                one big dataset or not. User have to specify either
                this parameter or split, not both. Defaults to False.
        """
        super().__init__()
        if split is not None and concat_subsets:
            raise ValueError("Either split of concat_dataset should be selected")

        self.split = split
        self.concat_subsets = concat_subsets

    def __call__(self, path: str) -> Iterable[dict[str, Any]]:
        if os.path.isfile(path):
            ds = datasets.Dataset.from_file(path)
        else:
            ds = datasets.load_from_disk(path)

        if self.split is not None:
            ds = ds[self.split]

        if self.concat_subsets and isinstance(ds, datasets.DatasetDict):
            ds = datasets.concatenate_datasets([ds[key] for key in ds.keys()])

        return ds


class WriteToArrow(PipelineTransformBase):
    is_materialized = True
    is_blocking = True

    def __init__(self, schema: pa.schema, path: str, buffer_size: int):
        self.schema = schema
        self.path = path
        self.buffer_size = buffer_size

    def __call__(self, input: Iterable[dict[str, Any]]) -> str:
        with pa.OSFile(self.path, mode="wb") as file:
            with pa.ipc.new_stream(file, schema=self.schema) as writer:
                for batch in batched(input, self.buffer_size):
                    flattened_batch = flatten_batch_dict(batch)

                    arrays = []
                    for name, dtype in zip(self.schema.names, self.schema.types):
                        arrays.append(pa.array(flattened_batch[name], type=dtype))

                    arrow_batch = pa.record_batch(
                        arrays,
                        names=self.schema.names,
                    )
                    writer.write(arrow_batch)

        return self.path


class LogToArrow(PipelineTransformBase):
    """Non blocking version of the WriteToArrow which just write
    inputs to arrow file and yield it to the next steps
    """

    def __init__(self, schema: pa.schema, path: str, buffer_size: int):
        self.schema = schema
        self.path = path
        self.buffer_size = buffer_size

    def __call__(self, input: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        with pa.OSFile(self.path, mode="wb") as file:
            with pa.ipc.new_stream(file, schema=self.schema) as writer:
                for batch in batched(input, self.buffer_size):
                    flattened_batch = flatten_batch_dict(batch)

                    arrays = []
                    for name, dtype in zip(self.schema.names, self.schema.types):
                        arrays.append(pa.array(flattened_batch[name], type=dtype))

                    arrow_batch = pa.record_batch(
                        arrays,
                        names=self.schema.names,
                    )
                    writer.write(arrow_batch)

                    for row in batch:
                        yield row


# Hack from https://github.com/huggingface/datasets/issues/6194#issuecomment-1708080653/
class _DatasetGeneratorPickleHack:
    def __init__(self, generator, generator_id=None):
        self.generator = generator
        self.generator_id = (
            generator_id if generator_id is not None else str(uuid.uuid4())
        )

    def __call__(self, *args, **kwargs):
        return self.generator(*kwargs, **kwargs)

    def __reduce__(self):
        return (_DatasetGeneratorPickleHack_raise, (self.generator_id,))


def _DatasetGeneratorPickleHack_raise(*args, **kwargs):
    raise AssertionError("cannot actually unpickle _DatasetGeneratorPickleHack!")


class WriteGeneratedDataset(PipelineTransformBase):
    is_materialized = True
    is_blocking = True

    def __init__(self, out_path: str, label2id_path: str) -> None:
        self.out_path = out_path
        self.label2id_path = label2id_path

    def read_label_to_id_file(self, label2id_path) -> None:
        with open(label2id_path, mode="r") as file:
            label2id = json.load(file)["label2id"]
            self.label2id = defaultdict(lambda: label2id["O"])
            self.label2id |= label2id

        class_names = [
            k for k, _ in sorted(self.label2id.items(), key=lambda item: item[1])
        ]
        ner_tags_feature = datasets.Sequence(
            feature=datasets.ClassLabel(
                num_classes=len(self.label2id), names=class_names
            )
        )
        self.ner_ds_features = datasets.Features(
            {
                "tokens": datasets.Sequence(feature=datasets.Value("string")),
                "ner_tags": ner_tags_feature,
            }
        )

    def map_label_to_id(self, labels: list[str]) -> list[int]:
        return [self.label2id[label] for label in labels]

    def __call__(self, input: Iterable[dict[str, Any]]) -> str:
        # since label2id is written by lazy LoadHFDataset transform we need to
        # take the first elem to be sure that file exists
        iter = primed(input)
        self.read_label_to_id_file(self.label2id_path)

        def dsrow_gen():
            for row in iter:
                tokens = row["tokens"]
                labels = row["labels"]
                ner_tags = self.map_label_to_id(labels)
                yield {"tokens": tokens, "ner_tags": ner_tags}

        ds = datasets.Dataset.from_generator(
            _DatasetGeneratorPickleHack(dsrow_gen), features=self.ner_ds_features
        )
        ds.save_to_disk(self.out_path)

        for file in glob(f"{self.out_path}/cache-*.arrow"):
            os.remove(file)

        return self.out_path
