"""This module contains translation pipeline transforms which use
either Fairseq or HF models to perform translation"""

from typing import Any, Iterable

from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM

from src.pipelines.transforms_base import PipelineTransformBase, batched, extract_key
from src.utils.model_context import use_hf_pipeline


class TransformersTranslationTransform(PipelineTransformBase):
    "Translation transfrom which use the given HF model"

    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        batch_size: int,
        device: int,
        column_key: str,
        model_path: str,
        src_lang_code: str | None = None,
        tgt_lang_code: str | None = None,
    ) -> None:
        super().__init__()
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.batch_size = batch_size
        self.device = device
        self.column_key = column_key
        self.model_path = model_path
        self._src_code = src_lang_code if src_lang_code else src_lang
        self._tgt_code = tgt_lang_code if tgt_lang_code else tgt_lang

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
            out_key = f"{self.tgt_lang}_translation"

            for batch in tqdm(batched(input, self.batch_size)):
                sentences = extract_key(batch, self.column_key)
                translation_iter = pipe(
                    sentences, src_lang=self._src_code, tgt_lang=self._tgt_code
                )

                for translation, in_row in zip(translation_iter, batch):
                    in_row[out_key] = translation[0]["translation_text"]
                    yield in_row
