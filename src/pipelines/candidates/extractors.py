"""This module contains different methods to
extract entity candidates in the target sentence
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Iterable, Tuple

from nltk import RegexpParser, Tree

from src.pipelines.transforms_base import PipelineTransformBase, batched, extract_key

logger = logging.getLogger(__file__)


class CandidateExtractorBase(ABC):
    def init(self) -> None:
        pass

    def deinit(self) -> None:
        pass

    @abstractmethod
    def extract(
        self,
        tgt_words: list[str],
        src_entities: list[dict[str, Any]] | None = None,
        **kwargs
    ) -> list[tuple[int, int]]:
        pass

    def extract_batched(
        self,
        tgt_words_batch: list[list[str]],
        src_entities_batch: list[list[str]],
        kwargs: list[dict[Any]],
    ) -> Iterable[list[tuple[int, int]]]:
        for words, src_entities, args in zip(
            tgt_words_batch, src_entities_batch, kwargs
        ):
            yield self.extract(words, src_entities, **args)


class CandidateExtractionTransform(PipelineTransformBase):
    def __init__(
        self,
        extractor: CandidateExtractorBase,
        input_words_key: str,
        batch_size: int = 1,
    ) -> None:
        self.extractor = extractor
        self.column_key = input_words_key
        self.batch_size = batch_size

    def __call__(self, input: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        self.extractor.init()

        if self.batch_size > 1:
            for batch in batched(input, self.batch_size):
                tgt_words = list(extract_key(batch, self.column_key))
                src_entities = list(extract_key(batch, "entities"))
                candidates_batch = self.extractor.extract_batched(
                    tgt_words, src_entities, batch
                )

                for row, candidates in zip(batch, candidates_batch):
                    row["tgt_candidates"] = candidates
                    yield row
        else:
            for row in input:
                tgt_words = row[self.column_key]
                src_entities = row["entities"]
                candidates = self.extractor.extract(tgt_words, src_entities, **row)

                row["tgt_candidates"] = candidates
                yield row

        self.extractor.deinit()


class DummySubrangeExtractor(CandidateExtractorBase):
    """Returns as candidates all subranges which contains at least
    min_words (default 1) and at max max_words (default number of input words).
    Ignores all subranges that contains a word from the set of stop_words
    """

    def __init__(
        self,
        min_words: int = 1,
        max_words: int | None = None,
        stop_words: set[str] | list[str] | None = {"!", "?"},
    ) -> None:
        super().__init__()
        assert min_words >= 1

        self.min_words = min_words
        self.max_words = max_words

        if stop_words and isinstance(stop_words, list):
            stop_words = set(stop_words)
        self.stop_words = stop_words

    def extract(
        self,
        tgt_words: list[str],
        src_entities: list[dict[str, Any]] | None = None,
        **kwargs
    ) -> list[tuple[int, int]]:
        N = len(tgt_words)
        max_words = self.max_words if self.max_words else N
        max_words += 1
        candidates = []

        for s in range(N - self.min_words + 1):
            max_end = min(s + max_words, N + 1)
            for e in range(s + self.min_words, max_end):
                if self.stop_words:
                    if not bool(self.stop_words.intersection(tgt_words[s:e])):
                        candidates.append((s, e))
                else:
                    candidates.append((s, e))

        return candidates


def extract_subranges(candidates: list[tuple[int, int]]) -> list[tuple[int, int]]:
    extended_candidates = []
    for cand in candidates:
        for start in range(cand[0], cand[1]):
            for end in range(start + 1, cand[1] + 1):
                extended_candidates.append((start, end))
    return extended_candidates


class POSRegexpChunkExtractor(CandidateExtractorBase):
    """Take as an input POS tags provided by, e.g. token classification model
    and retieve candidates based on them according to specified context-free grammar"""

    def __init__(
        self,
        grammar: str,
        extract_subranges: bool = False,
        candidate_terminal: str = "CAND",
    ) -> None:
        super().__init__()

        self.chunker = RegexpParser(grammar)
        self.extract_subranges = extract_subranges
        self.candidate_terminal = candidate_terminal

    def chunk_tree_to_candidates(self, chunk_tree: Tree) -> list[Tuple[int, int]]:
        shift = 0  # number of words before the current node
        candidates = []
        for node in chunk_tree:
            if isinstance(node, tuple):  # just a tagged word
                shift += 1
                continue
            elif node.label() == self.candidate_terminal:
                candidates.append(shift, shift + len(node.leaves()))
            shift += len(node.leaves())

        return candidates

    def extract(
        self, tgt_words: list[str], src_entities: list[dict[str, Any]], **kwargs
    ) -> list[tuple[int, int]]:
        pos_labels = kwargs.pop("pos_labels")

        if len(pos_labels) != len(tgt_words):
            logger.warning("Number of POS tags and tgt_words are different!")

        tagged_words = list(zip(tgt_words, pos_labels))
        chunk_tree = self.chunker(tagged_words)
        candidates = self.chunk_tree_to_candidates(chunk_tree)

        if self.extract_subranges:
            candidates = extract_subranges(candidates)

        return candidates


class CandidateNERExtractor(CandidateExtractorBase):
    """Converts output of a NER model which predicts B-CAND and I-CAND label for
    original sentence in the target language (see src.models.ner.candidates
    module) to a target candidates format"""

    def __init__(
        self,
        entities_key: str = "entities",
        extract_subranges: bool = False,
    ) -> None:
        super().__init__()

        self.column_key = entities_key
        self.extract_subrange = extract_subranges

    def extract(
        self,
        tgt_words: list[str],
        src_entities: list[dict[str, Any]] | None = None,
        **kwargs
    ) -> list[tuple[int, int]]:
        entities_candidates = kwargs.pop(self.column_key)
        candidates = list(
            map(lambda e: (e["start_idx"], e["end_idx"]), entities_candidates)
        )

        if self.extract_subrange:
            candidates = extract_subranges(candidates)

        return candidates
