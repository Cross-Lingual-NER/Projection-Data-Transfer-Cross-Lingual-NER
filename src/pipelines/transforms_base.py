"""This module contains a base class for a pipeline tranform (step)
and some simple useful utitlities and tranfroms"""

from abc import ABC, abstractmethod
from itertools import islice
from typing import Any, Generator, Iterable, Tuple


class PipelineTransformBase(ABC):
    is_materialized = False
    is_blocking = False

    @abstractmethod
    def __call__(self, input: Any) -> Any:
        pass


def batched(iterable: Iterable[Any], batch_size) -> Iterable[list[Any]]:
    it = iter(iterable)

    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch


def unbatched(iterable: Iterable[list[Any]]) -> Iterable[Any]:
    for batch in iterable:
        for row in batch:
            yield row


def extract_key(
    iterable: Iterable[dict[str, Any]], key: str
) -> Generator[Any, None, None]:
    for row in iterable:
        yield row[key]


def flatten_batch_dict(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {k: [dic[k] for dic in batch] for k in batch[0]}


def unflatten_batch_dict(batch: dict[str, list[Any]]) -> list[dict[str, Any]]:
    res = [{}] * len(batch[list(batch.keys())[0]])
    for key, val_list in batch:
        for i, val in enumerate(val_list):
            res[i][key] = val
    return res


class MergeTransform(PipelineTransformBase):
    def __call__(
        self, input: Tuple[Iterable[dict[str, Any]]]
    ) -> Iterable[dict[str, Any]]:
        for row_tuple in zip(*input):
            res = row_tuple[0]
            for row in row_tuple[1:]:
                for key, val in row.items():
                    res[key] = val
            yield res


class RemoveKeysTransform(PipelineTransformBase):
    def __init__(self, keys: list[str]) -> None:
        self.keys = keys

    def __call__(self, input: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        for row in input:
            for key in self.keys:
                row.pop(key, None)
            yield row


class RenameTransform(PipelineTransformBase):
    def __init__(self, keys_map: dict[str, str]) -> None:
        self.keys_map = keys_map

    def __call__(self, input: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        for row in input:
            for old_key, new_val in self.keys_map.items():
                row[new_val] = row.pop(old_key, None)
            yield row


class CachedTransform(PipelineTransformBase):
    def __init__(self, cache: Any) -> None:
        super().__init__()
        self.cache = cache

    def __call__(self, input: Any) -> Any:
        return self.cache
