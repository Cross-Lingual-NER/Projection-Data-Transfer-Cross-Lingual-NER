"""This module contains implementation of the
pipeline step which performs POS labeling of the
input words"""

from typing import Any, Iterable

from tqdm import tqdm
from transformers import AutoModelForTokenClassification

from src.pipelines.transforms_base import PipelineTransformBase, batched, extract_key
from src.utils.model_context import use_hf_pipeline


class POSTransform(PipelineTransformBase):
    """POS transform which use specified POS HF Transfrormers model to
    label every words with the corresponding POS tags. Expects as an input
    already splitted into words sentence and doesn't work with a whole
    sentence. Outputs POS tag by setting pos_labels key in the input sample"""

    def __init__(
        self,
        model_path: str,
        batch_size: int = 32,
        device: int = 0,
        column_key: str = "words",
    ) -> None:
        super().__init__()

        self.column_key = column_key
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size

    def __call__(self, input: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        with use_hf_pipeline(
            "tokenwise-classification",
            self.model_path,
            pipeline_kwargs={
                "device": self.device,
                "batch_size": self.batch_size,
            },
            AutoModelClass=AutoModelForTokenClassification,
        ) as pipe:
            for batch in tqdm(batched(input, self.batch_size)):
                tokens = extract_key(batch, self.column_key)
                pos_out_iter = pipe(tokens)

                for pos_out, in_row in zip(pos_out_iter, batch):
                    pos_labels = []
                    for word_pred in pos_out:
                        pos_labels.append(word_pred["entity_group"])

                    in_row["pos_labels"] = pos_labels
                    yield in_row
