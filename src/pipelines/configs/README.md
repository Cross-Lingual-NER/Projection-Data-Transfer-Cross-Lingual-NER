# Pipeline configuration files

To handle all complexity of the XLNER pipelines configurations we use the
[_hydra_](https://hydra.cc/docs/1.0/intro/) package which allows us to reuse a lot of shared configs as well
as split and efficiently compose them based on the semantic of every particular configuration property
(e.g. paths are independent from pipelines, etc).

The main file which references all subconfigs is `default.yaml`. It contains high-level configs, the
actual configs for every particular logic group can be found in the subdirectories, in particular:
- paths - configs related to paths where translation models, data, output directory of the pipeline are located
- runner - configs that specify which runner (see `src/pipelines/runners.py`) to use to run a pipeline
- schemas - files which contain arrow schemas specification that are shared among many pipelines (just to avoid copying)
- pipeline - configs which contains pipeline description as it is written in the README file in the root of this repository
- experiments - whereas configs from the `pipeline` folder specify pipelines in general and left some parameters unspecified,
the main idea of experiments is to take a pipeline and fix/assign parameters of this pipeline, i.t. implementation of the pipeline with fixed
translation model architecture, target and source language, etc.

Hydra supports having several options of configs for the same high-level configuration property. In order to specify which config to use
one have to reference the desired config file with their name, e.g. if one want to use `pipeline/align/src2tgt_eval_compare.yaml` config for as a pipeline, they should
reference it as `pipeline=align/src2tgt_eval_compare`, i.e. without extension and . Information about all of this as well as hot to override configurations from a command line
can be found in the [_hydra_ docs](https://hydra.cc/docs/1.0/intro/), so we ask you **carefully read it**.

The structure of the pipeline config directory is the following:
- different approaches grouped in the subdirectories, e.g. crop, simalign, etc
- if approach supports different word to word alignment strategies (src2tgt and tgt2tgt) it will be added to the name of the pipeline as a suffix or mentioned as `backtrans`

Experiments configs inherit the structure of the pipeline configs, but since they are more specific than pipelines have more complicated names in particular adds
as a suffix a dataset name (with optionally language), translation / NER model name, etc to the name of the pipeline config. The pipeline, config refers to, **is always written in the defaults**, but ignored by current _hydra_ version, so have to be specified manually.