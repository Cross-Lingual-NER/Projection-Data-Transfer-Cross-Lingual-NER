"""This module contatins implementation of the pipeline transform
that project entities based on word to word alignments
"""

import logging
from typing import Any, Iterable

from src.pipelines.align.matching import match_entities_based_on_word_alignments
from src.pipelines.transforms_base import PipelineTransformBase

logger = logging.getLogger(__file__)


class AlignedEntityProjectionTransform(PipelineTransformBase):
    """Based on word to word alignments obtained from WordAlignTransform
    project entitites from source to target sentence"""

    def __init__(
        self,
        input_orig_words_key: str,
        input_trans_words_key: str,
        length_ratio_threshold: float,
        merge_distance: int = 1,
    ) -> None:
        self.orig_key = input_orig_words_key
        self.trans_key = input_trans_words_key
        self.length_ratio_threshold = length_ratio_threshold
        self.merge_distance = merge_distance

    def __call__(self, input: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        for row in input:
            orig_words = row[self.orig_key]
            trans_words = row[self.trans_key]
            entities = row["entities"]
            alignments = row["word_alignments"]

            try:
                labels = match_entities_based_on_word_alignments(
                    orig_words,
                    trans_words,
                    entities,
                    alignments,
                    self.length_ratio_threshold,
                    self.merge_distance,
                )
                row["labels"] = labels
            except Exception:
                logger.warn(f"Projection failed for the input: {row}")
                row["labels"] = ["O"] * len(orig_words)

            yield row
