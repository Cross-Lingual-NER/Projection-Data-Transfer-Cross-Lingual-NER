"""This module contains functions which performs entity projection
based on word to word alignemnts
"""


def match_entities_based_on_word_alignments(
    original_words: list[str],
    translated_words: list[str],
    entities: list[dict],
    word_alignments: list[(int, int)],
    length_ratio_threshold: float = 0.8,
    merge_distance: int = 1,
) -> list[str]:
    """Perform matching of entities in the original
    and translated sentence based on given word2word alignments.
    Return the list of labels for original
    words in the IOB2 format.

    Args:
        original_words (list[str]): words of the original sentence
        translated_words (list[str]): words of the translated sentence
        entities (list[dict]): label, start and end index in the
            translated words of entity
        word_alignments (list[Tuple[int, int]]): word to word alignments
            between original and translated words
        length_ratio_threshold (float): entity will be matched iff
            a ratio of the projected length and given entity length is higher
            than this value
        merge_distance (int): word to word alignments can be not perfect, by setting
            this argument you can ask a function to fill gaps in aligments by
            merging all target entity candidate if distance between them less than
            provided threshold

    Returns:
        list[str]: list of labeles in IOB2 format for original words
    """

    labels = ["O" for _ in original_words]  # initially no entities
    alignments_by_trans_words = [
        [align[0] for align in word_alignments if align[1] == i]
        for i in range(len(translated_words))
    ]  # list of aligned original word for every translated word

    for entity in entities:
        start_idx, end_idx = entity["start_idx"], entity["end_idx"]
        label = entity["label"]
        len_entity = end_idx - start_idx

        candidates = generate_entity_candidates_from_alignments(
            start_idx, end_idx, alignments_by_trans_words
        )

        if merge_distance > 0 and len(candidates) > 0:
            candidates = merge_adjacent_candidates(
                candidates, max_distance=merge_distance
            )

        # Filter out small (ratio of lenght between source entity and candidate is
        # below threshold) entity candidates which was obtained because of
        # wrong alignments
        projected_entities = filter(
            lambda c: (c[1] - c[0]) / len_entity > length_ratio_threshold, candidates
        )

        # label words
        for ent in projected_entities:
            labels[ent[0]] = "B-" + label
            for i in range(ent[0] + 1, ent[1]):
                labels[i] = "I-" + label

    return labels


def generate_entity_candidates_from_alignments(
    start_idx: int, end_idx: int, alignments_by_trans_words: list[list[int]]
) -> list[tuple[int, int]]:
    """Returns entity candidates in the target sentence. As entity candidate we
    consider any continious range of target words which are aligned to any word
    in the source sentence"""
    candidates = []
    for idx in range(start_idx, end_idx):
        # idxs of all correspondence in original words
        orig_corrs = alignments_by_trans_words[idx]

        for corr_idx in orig_corrs:
            new_candidate = True
            for candidate in candidates:
                if candidate[0] <= corr_idx < candidate[1]:  # inside candidate
                    new_candidate = False
                elif corr_idx == candidate[1]:  # candidate + 1 word right
                    candidate[1] += 1
                    new_candidate = False
                elif corr_idx == candidate[0] - 1:  # 1 word left + candidate
                    candidate[0] -= 1
                    new_candidate = False
            if new_candidate:
                candidates.append([corr_idx, corr_idx + 1])
    return candidates


def merge_adjacent_candidates(
    candidates: list[tuple[int, int]], max_distance: int = 1
) -> list[tuple[int, int]]:
    """Merge candidates if a distance between them less than threshold. Helps to
    fill gaps in aligments"""

    candidates = sorted(candidates, key=lambda cand: cand[0])
    merged_candidates = [candidates[0]]

    for cand in candidates[1:]:
        if cand[0] - merged_candidates[-1][1] > max_distance:
            merged_candidates.append(cand)
        else:
            merged_candidates[-1][1] = cand[1]  # extend previous candidate

    return merged_candidates
