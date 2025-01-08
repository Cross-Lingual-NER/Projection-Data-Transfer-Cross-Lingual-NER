#!/usr/bin/env python
"""This script download a parallel corpus from HF hub and
preprocess it in order to use it for training awesome-align model
"""

import argparse
from functools import partial

import datasets
import nltk
import numpy as np


def save_to_file(row, file):
    print(row["line"], file=file)


def merge_translations(row):
    res = []
    for _, sent in row["translation"].items():
        words = nltk.word_tokenize(sent)
        sent = " ".join(words)
        res.append(sent)
    out = " ||| ".join(res)
    return {"line": out}


def load_dataset(args: argparse.Namespace) -> None:
    subset = f"{args.lang1}-{args.lang2}"
    ds = datasets.load_dataset(args.dataset, subset)["train"]

    filtered_ds = ds.filter(lambda row: len(row["translation"]["en"]) > args.min_length)

    idxs = np.random.randint(0, len(filtered_ds), size=args.sample_size)
    align_ds = filtered_ds.select(idxs)

    align_ds = align_ds.map(merge_translations, remove_columns="translation")

    with open(args.out_path, mode="w", encoding="utf-8") as file:
        save_func = partial(save_to_file, file=file)
        align_ds.map(save_func)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and preprocess europarl-ner dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Helsinki-NLP/europarl",
        help="Translation dataset from the HF hub",
    )
    parser.add_argument("--lang1", type=str, default="en", help="First language code")
    parser.add_argument("--lang2", type=str, default="it", help="Second language code")
    parser.add_argument(
        "--min_length",
        type=int,
        default=30,
        help="Minimum lenght of the sentence in characters",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=500000,
        help="Size of the sample to be sampled from the dataset",
    )
    parser.add_argument("out_path", type=str, help="path to the output file")
    args = parser.parse_args()

    load_dataset(args)
