#!/usr/bin/env python3 -u
"""This script retieves the label2id dictionary from the specified HF model
for token classification"""

import argparse
import json

from transformers import AutoConfig


def save_label2id_from_model(model_path: str, save_path: str) -> None:
    config = AutoConfig.from_pretrained(model_path)
    with open(save_path, "w") as out:
        json.dump({"label2id": config.label2id}, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get label2id from model config and save it to json file"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path (or name on HF Hub) to the NER model",
    )
    parser.add_argument(
        "--save_path",
        required=True,
        type=str,
        help="Output path for label2id dict (json file)",
    )
    args = parser.parse_args()

    save_label2id_from_model(args.model_path, args.save_path)
