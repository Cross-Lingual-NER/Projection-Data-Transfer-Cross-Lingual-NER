"""This module contains pipeline runners. Pipeline runner is an
abstraction which allows user implement their own logic for executing
pipelines, e.g. in single thread, using multiprocessing, internode
communication, etc"""

from abc import ABC, abstractmethod
from collections import namedtuple
from itertools import tee
from typing import Any

from src.pipelines.transforms_base import PipelineTransformBase

Transform = namedtuple("Transform", ["name", "deps", "transform"])


class PipelineRunnerBase(ABC):
    @abstractmethod
    def init_pipeline(self, transforms: list[Transform]):
        "Prepare everything that pipeline runner need to work"
        pass

    @abstractmethod
    def get_inputs_for_step(self, transform: Transform) -> list[Any]:
        "Return stored internally inputs for the specified step"
        pass

    @abstractmethod
    def run_step(self, step: PipelineTransformBase, input: Any) -> None:
        "Actually execute (usually just start) the given step"
        pass


def calculate_num_copies(transforms: list[Transform]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trans in transforms:
        counts[trans.name] = 0

    for trans in transforms:
        for dep in trans.deps:
            counts[dep] += 1

    return counts


class SingleThreadRunner(PipelineRunnerBase):
    "Run pipeline steps sequentially in the single thread"

    def __init__(self) -> None:
        super().__init__()

    def init_pipeline(self, transforms: list[Transform]):
        self.copy_counts = calculate_num_copies(transforms)
        self.iterators = {k: [] for k in self.copy_counts}

    def get_inputs_for_step(self, transform: Transform) -> list[Any]:
        inputs = []
        for dep in transform.deps:
            inputs.append(self.iterators[dep].pop())
        return inputs

    def run_step(self, transform: Transform, inputs: Any) -> None:
        iterator = transform.transform(inputs)

        n = self.copy_counts[transform.name]
        if n > 1:
            if transform.transform.is_materialized:
                self.iterators[transform.name].extend([iterator] * n)
            else:
                self.iterators[transform.name].extend(list(tee(iterator, n)))
        else:
            self.iterators[transform.name].append(iterator)
