"""After backtranslation step one should extract entities from
backtranslated sentence. This process involves splitting sentence word
by word. By since splitting of some punctuation signs / words is
ambiguous and we want to test our pipeline by comparing with already
splitted and labeled into words sentences (such as wikiann dataset)
we need to implement different word tokenization approaches. This
file contains them as well as pipeline transform to do it.
"""

import re
from abc import abstractmethod
from itertools import groupby
from typing import Any, Iterable

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tokenizers.pre_tokenizers import BertPreTokenizer, PreTokenizer, WhitespaceSplit

try:
    import jieba  # install if you need a ChineseTokenizerp
except ImportError:
    pass

from src.pipelines.transforms_base import PipelineTransformBase


class WordSplitterBase:
    robust_slots_splitting: bool = False

    @staticmethod
    def __find_all_slots(word: str):
        i = word.find("__SLOT")
        while i != -1:
            j = word.find("__", i + 1)
            if j != -1:
                yield i, j + 2
                i = word.find("__SLOT", j + 1)
            else:
                return i, -1

    @staticmethod
    def __handle_slots(splitted_words: Iterable[str]) -> Iterable[str]:
        # handling of adjacent slots of SLOT<symbol>SLOT
        prev_is_incomplete = False
        for word in splitted_words:
            if prev_is_incomplete:
                prev_is_incomplete = False
                if word == "__":
                    continue

            prev_end = 0
            for start, end in WordSplitterBase.__find_all_slots(word):
                if end == -1:  # if slot is incomplete: __SLOT6 __
                    prev_is_incomplete = True
                    yield word + "__"
                else:
                    if prev_end < start:
                        yield word[prev_end:start]
                    yield word[start:end]
                    prev_end = end
            if prev_end < len(word):
                yield word[prev_end:]

    @abstractmethod
    def split(self, sent: str, **kwds: Any) -> Iterable[str]:
        pass

    def __call__(self, sent: str, **kwds: Any) -> list[str]:
        """The main routine which splits sentence into words.

        Args:
            sent (str): input sentence

        Returns:
            list[str]: list of words of the given sentence
        """
        words = list(self.split(sent, **kwds))

        if self.robust_slots_splitting:
            words = self.__handle_slots(words)

        return words


class ChineseSplitter(WordSplitterBase):
    def split(self, sent: str, **kwds: Any) -> Iterable[str]:
        tokens = jieba.cut(sent)
        words = [tok for tok in tokens if tok.strip() != ""]
        return words


class HFSplitterBase(WordSplitterBase):
    def __init__(self, pretokenizer: PreTokenizer) -> None:
        super().__init__()
        self.pretokenizer = pretokenizer

    def split(self, sent: str, **kwds: Any) -> Iterable[str]:
        return map(lambda w: w[0], self.pretokenizer.pre_tokenize_str(sent))


class WhitespaceSplitter(HFSplitterBase):
    def __init__(self) -> None:
        super().__init__(WhitespaceSplit())


class HFBertSplitter(HFSplitterBase):
    def __init__(self) -> None:
        super().__init__(BertPreTokenizer())


class NLTKSplitter(WordSplitterBase):
    def __init__(self) -> None:
        nltk.download("punkt")

    def split(self, sent: str, **kwds: Any) -> Iterable[str]:
        return word_tokenize(sent, **kwds)


class WhitespaceLessLanguageSplitter(WordSplitterBase):
    def split(self, sent: str, **kwds: Any) -> Iterable[str]:
        yield from sent


class JapaneseThaiSplitter(WordSplitterBase):
    """Based on https://github.com/edchengg/easyproject
    which is avaliable under MIT License"""

    FULL2HALF = dict((i + 0xFEE0, i) for i in range(0x30, 0x40))

    def halfen(self, s: str) -> str:
        """
        Convert full-width characters to ASCII counterpart.
        halfen('１２３４５６７８９０') == '1234567890'
        """
        return str(s).translate(self.FULL2HALF)

    def judge_if_Thai_or_Japanese_char(self, sent):
        return [
            (
                1
                if re.match(
                    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f\u0E00-\u0E7Fa]+",  # noqa: E501
                    i,
                )
                is not None
                else 0
            )
            for i in sent
        ]

    def merge_adjacent_identical_numbers(self, number_list):
        U = []
        for _, group in groupby(number_list):
            U.append(list(group))

        begin_end_idx_list = []
        count = 0
        for i in U:
            start_idx = count
            for j in i:
                count += 1
            end_idx = count
            begin_end_idx_list.append([start_idx, end_idx, i[0]])

        return U, begin_end_idx_list

    def split(self, sent: str, **kwds: Any) -> Iterable[str]:
        words = []

        Chinese_or_Eng_list = self.judge_if_Thai_or_Japanese_char(sent)
        _, group = self.merge_adjacent_identical_numbers(Chinese_or_Eng_list)

        for start_idx, end_idx, Chinese_or_not in group:
            # if English, use word_tokenize
            if Chinese_or_not == 0:
                char_list = sent[start_idx:end_idx]
                tokenized_list = word_tokenize(char_list)
                change_dict = {
                    "``": '"',
                    "''": '"',
                }
                # need to convert full-width number in CoNLL data to regular
                # half-width number
                tokenized_list = [
                    (
                        self.halfen(i)
                        if i not in change_dict
                        else self.halfen(change_dict[i])
                    )
                    for i in tokenized_list
                ]
                words.extend(tokenized_list)
            # if Chinese, do char-tokenization
            elif Chinese_or_not == 1:
                # need to convert full-width number in CoNLL data to regular
                # half-width number
                words.extend([self.halfen(c) for c in sent[start_idx:end_idx]])

        return words


class WikiannSplitter(WordSplitterBase):
    """Special case of splitter for Wikiann dataset (especially HSB part).
    We need it for evaluation of our pipeline because the merged and then splitted
    original sentence from this dataset should match the original ones
    (has the same length and words) in order to compute seqeval metrics.

    Expect that original dataset has been converted (prettified)
    by the provided function
    """

    def __init__(self) -> None:
        self._main_splitter = NLTKSplitter()

    @staticmethod
    def adjust_quotes(words):
        adjusted_words = []
        for word in words:
            if word == "``" or word == "''":
                adjusted_words.append('"')
            else:
                adjusted_words.append(word)
        return adjusted_words

    @staticmethod
    def prettify_original_ds(row: dict[str, Any]) -> dict[str, list[Any]]:
        """Function to be applied to the original dataset via map method.
        Converts quotation marks and some special cases
        to the fixed predefined format and respectively adjust labels.

        Args:
            row (dict[str, Any]): row of the Wikiann dataset

        Returns:
            dict[str, list[Any]]: converted tokens and corresponding labels
        """

        tokens = row["tokens"]
        labels = row["ner_tags"]

        res_tokens = []
        res_labels = []

        predefined_map = {
            # train split
            "swj.": [("swj", -1), (".", 0)],
            "*1859-†1917": [("1859-1917", -1)],
            # valid split
            "******": [],
            "****": [],
            "II.": [("II", -1), (".", 0)],
            "Swj.": [("Swj", -1), (".", 0)],
            "[ghwino]": [("[", 0), ("ghwino", -1), ("]", 0)],
            # test split
            "***": [],
            "*****": [],
            "**": [],
            "*1970": [("1970", -1)],
        }

        for token, label in zip(tokens, labels):
            if token[:2] == "''":
                if len(res_tokens) > 0 and res_tokens[-1] == "'":
                    res_tokens.pop()
                    res_labels.pop()
                    res_tokens.append('"')
                    res_labels.append(0)
                else:
                    res_tokens.append('"')
                    res_labels.append(0)
                if len(token) > 2:
                    res_tokens.append(token[2:])
                    res_labels.append(label)
            elif len(res_tokens) > 0 and res_tokens[-1] == '"' and token == "'":
                continue
            else:
                token = token.removesuffix("“")
                if token in predefined_map:
                    for t, l in predefined_map[token]:
                        res_tokens.append(t)
                        res_labels.append(label if l == -1 else l)
                else:
                    res_tokens.append(token)
                    res_labels.append(label)

        return {"tokens": res_tokens, "ner_tags": res_labels}

    def split(self, sent: str, **kwds: Any) -> Iterable[str]:
        words = self._main_splitter(sent)
        return self.adjust_quotes(words)


class WordSplitTransform(PipelineTransformBase):
    def __init__(
        self,
        word_splitter: WordSplitterBase,
        sent_column_key: str,
        out_key: str = "tokens",
        check_merged_slots: bool = False,
    ) -> None:
        self.sent_column_key = sent_column_key
        self.out_key = out_key
        self.splitter = word_splitter
        self.splitter.robust_slots_splitting = check_merged_slots

    def __call__(self, input: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        for row in input:
            words = self.splitter(row[self.sent_column_key])
            row[self.out_key] = words
            yield row


class DetokenizeTransform(PipelineTransformBase):
    """Merge tokens into sentences"""

    def __init__(
        self,
        tokens_key: str = "tokens",
        out_key: str = "tgt_text",
        lang_has_whitespace: bool = True,
    ) -> None:
        super().__init__()
        self.tokens_key = tokens_key
        self.out_key = out_key
        self.lang_has_whitespaces = lang_has_whitespace
        if lang_has_whitespace:
            self.detokenizer = TreebankWordDetokenizer()

    def __call__(self, input: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        for row in input:
            if self.lang_has_whitespaces:
                row[self.out_key] = self.detokenizer.detokenize(row[self.tokens_key])
            else:
                row[self.out_key] = "".join(row[self.tokens_key])

            yield row
