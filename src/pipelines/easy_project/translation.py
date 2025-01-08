from typing import Any, Iterable

from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, Pipeline

from src.pipelines.easy_project.matching import match_entities, surround_entities
from src.pipelines.transforms_base import batched, extract_key
from src.pipelines.translation import TransformersTranslationTransform
from src.pipelines.word_splitting import WordSplitterBase
from src.utils.model_context import use_hf_pipeline


class EasyProjectBackTranslationTransform(TransformersTranslationTransform):
    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        batch_size: int,
        device: int,
        src_words_key: str,
        src_entities_key: str,
        model_path: str,
        word_splitter: WordSplitterBase,
        sim_threshold: float = 0.5,
        out_entities_key: str = "tgt_entities",
        out_words_key: str = "tgt_words",
        src_lang_code: str | None = None,
        tgt_lang_code: str | None = None,
        max_length: int = 400,
    ) -> None:
        super().__init__(
            src_lang,
            tgt_lang,
            batch_size,
            device,
            src_words_key,
            model_path,
            src_lang_code,
            tgt_lang_code,
        )
        self.word_splitter = word_splitter
        self.sim_threshold = sim_threshold
        self.max_length = max_length

        self.src_entities_key = src_entities_key
        self.out_entities_key = out_entities_key
        self.out_words_key = out_words_key

    @staticmethod
    def _extract_entities(
        src_words: list[str], entities: list[str, Any]
    ) -> Iterable[str]:
        detokenizer = TreebankWordDetokenizer()
        for ent in entities:
            start_idx = ent["start_idx"]
            end_idx = ent["end_idx"]

            yield detokenizer.detokenize(src_words[start_idx:end_idx])

    def _translate_entities(
        self,
        src_words_batch: list[list[str]],
        src_entities_batch: Iterable[list[str, Any]],
        pipe: Pipeline,
    ) -> Iterable[list[str]]:
        for src_words, src_entities in zip(src_words_batch, src_entities_batch):
            translated_entities = pipe(
                self._extract_entities(src_words, src_entities),
                src_lang=self._src_code,
                tgt_lang=self._tgt_code,
            )
            trans_text = map(
                lambda trans: trans[0]["translation_text"], translated_entities
            )
            yield list(trans_text)

    def _translate_sententeces(
        self,
        src_words: list[list[str]],
        src_entities: list[list[str, Any]],
        pipe: Pipeline,
    ) -> Iterable[str]:
        embraced_sent = [
            surround_entities(words, ents)
            for words, ents in zip(src_words, src_entities)
        ]
        translation_iter = pipe(
            embraced_sent,
            src_lang=self._src_code,
            tgt_lang=self._tgt_code,
            max_length=self.max_length,
        )
        return map(lambda trans: trans["translation_text"], translation_iter)

    def __call__(self, input: Iterable[dict[Any]]) -> Iterable[dict[Any]]:
        task = f"translation_{self.src_lang}_to_{self.tgt_lang}"
        with use_hf_pipeline(
            task,
            self.model_path,
            pipeline_kwargs={
                "device": self.device,
                "batch_size": self.batch_size,
            },
            AutoModelClass=AutoModelForSeq2SeqLM,
        ) as pipe:
            for batch in tqdm(batched(input, self.batch_size)):
                src_words_batch = list(extract_key(batch, self.column_key))
                src_entities_batch = list(extract_key(batch, self.src_entities_key))

                translated_entities = self._translate_entities(
                    src_words_batch, src_entities_batch, pipe
                )
                translation_iter = self._translate_sententeces(
                    src_words_batch, src_entities_batch, pipe
                )

                for trans_sent, trans_entities, src_entities, in_row in zip(
                    translation_iter, translated_entities, src_entities_batch, batch
                ):
                    tgt_words, tgt_entities = match_entities(
                        trans_sent,
                        trans_entities,
                        src_entities,
                        self.word_splitter,
                        self.sim_threshold,
                    )

                    in_row[self.out_words_key] = tgt_words
                    in_row[self.out_entities_key] = tgt_entities
                    yield in_row
