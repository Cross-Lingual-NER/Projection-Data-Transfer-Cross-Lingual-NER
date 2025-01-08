"""Given candidates we need to compute the compatibility
score between the source entities and target candidates. This modules
contains implementation of these evaluators
"""

from typing import Any

from scipy.sparse import csr_matrix


def get_entities_spans(entities: list[dict[str, Any]]) -> list[tuple[int, int]]:
    return [(e["start_idx"], e["end_idx"]) for e in entities]


def get_relative_lenght_alignment_scores(
    word_alignments: list[tuple[int, int]],
    src_entity_spans: list[tuple[int, int]],
    tgt_candidates: list[tuple[int, int]],
) -> csr_matrix:
    """Compute matching costs between source entities and target candidates
    using word to word alignments. This implementation computes scrores as
    a number of aligned word pairs between a source entity and candidate
    divided by sum of number of words in a source entity and a candidate.
    The main idea is that if all entity/candidates words are aligned that
    it is probably a coorrect matching

    Args:
        word_alignments (list[tuple[int, int]]): word to word alignemnt
            between target and source words
        src_entity_spans (list[tuple[int, int]]): spans (in word indices) of
            source entities
        tgt_candidates (list[tuple[int, int]]): spans of candidates

    Returns:
        csr_matrix: matrix of costs between source entities and candidates.
            We use csr format because matrix is usually sparse.
    """

    weights = []
    col_indices = []
    row_ptr = [0]

    n_words = max(src_entity_spans, key=lambda span: span[1])[1]

    alignments_by_src_words = [
        [align[0] for align in word_alignments if align[1] == i] for i in range(n_words)
    ]  # list of aligned target word for every source word

    for src_s, src_e in src_entity_spans:
        src_aligns = alignments_by_src_words[src_s:src_e]
        src_ent_len = src_e - src_s
        for col, (tgt_s, tgt_e) in enumerate(tgt_candidates):
            num_aligned_words = 0
            for tgt_idxs in src_aligns:
                num_aligned_words += sum(
                    1 if tgt_s <= idx < tgt_e else 0 for idx in tgt_idxs
                )

            if num_aligned_words > 0:
                tgt_cand_len = tgt_e - tgt_s
                w = 2 * num_aligned_words / (src_ent_len + tgt_cand_len)

                weights.append(w)
                col_indices.append(col)

        row_ptr.append(len(col_indices))

    mat_shape = (len(src_entity_spans), len(tgt_candidates))
    return csr_matrix((weights, col_indices, row_ptr), shape=mat_shape)
