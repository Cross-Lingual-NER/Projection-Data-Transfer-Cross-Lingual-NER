import re
from difflib import SequenceMatcher
from typing import Any

from nltk.tokenize.treebank import TreebankWordDetokenizer

from src.pipelines import word_splitting


def surround_entities(words: list[str], entities: list[dict[str, Any]]) -> str:
    out_words = []
    detokenizer = TreebankWordDetokenizer()

    prev_ent_word_idx = 0
    for ent in entities:
        ent_start = ent["start_idx"]
        ent_end = ent["end_idx"]

        out_words.extend(words[prev_ent_word_idx:ent_start])
        out_words.append("[")
        out_words.extend(words[ent_start:ent_end])
        out_words.append("]")

        prev_ent_word_idx = ent_end
    out_words.extend(words[prev_ent_word_idx:])

    return detokenizer.detokenize(out_words)


def similarity_score(span1: str, span2: str) -> float:
    return SequenceMatcher(None, span1.lower(), span2.lower()).ratio()


def match_entities(
    trans_sent: str,
    ent_translations: list[str],
    src_entities: list[dict[str, Any]],
    word_splitter: word_splitting.WordSplitterBase,
    sim_threshold: float = 0.5,
) -> tuple[list[str], list[dict[str, Any]]]:
    src_entities = src_entities.copy()
    tgt_entities = []
    tgt_words = []

    prev_char_idx = 0
    for match in re.finditer(r"\[(.*?)\]", trans_sent):
        max_score = 0
        max_idx = -1
        for idx, ent_trans in enumerate(ent_translations):
            score = similarity_score(match.group(), ent_trans)
            if score > sim_threshold and score > max_score:
                max_idx = idx

        if max_idx >= 0:
            start_idx, end_idx = match.span()
            label = src_entities[max_idx]["label"]

            prev_subsent = trans_sent[prev_char_idx:start_idx]
            tgt_words.extend(word_splitter(prev_subsent))

            ent_subsent = trans_sent[start_idx + 1 : end_idx - 1]  # ignore braces
            ent_start = len(tgt_words)
            tgt_words.extend(word_splitter(ent_subsent))
            ent_end = len(tgt_words)

            tgt_entities.append(
                {"start_idx": ent_start, "end_idx": ent_end, "label": label}
            )

            prev_char_idx = end_idx
            del ent_translations[max_idx]
            del src_entities[max_idx]

    tail_subsent = trans_sent[prev_char_idx:]
    tgt_words.extend(word_splitter.split(tail_subsent))

    return tgt_words, tgt_entities
