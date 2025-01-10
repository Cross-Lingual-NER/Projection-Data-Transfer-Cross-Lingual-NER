# Projection-based-Data-Transfer-approach-for-Cross-Lingual-NER
This GitHub repository provides the code and resources for our paper, accepted for presentation at the NoDaLiDa/Baltic-HLT 2025 main conference: Revisiting Projection-based Data Transfer for Cross-Lingual Named Entity Recognition in Low-Resource Languages

## Project Structure
------------

    ├── LICENSE
    ├── README.md
    ├── experiments_results <- CSV files containing full experiment results
    ├── slurm               <- SLURM jobs to run experiments from the paper
    ├── requirements.txt    <- Requirements file for reproducing the analysis environment
    ├── setup.py            <- Makes project pip installable (pip install -e .)
    └─── src                <- Source code for use in this project.
        ├── data            <- Scripts to download, preproces or generate data
        ├── models          <- Scripts to train models and infer models*
        │   └── ner         <- Named Entity Recognition models (based on HF Transformers)
        └── pipelines       <- XLNER pipelines, transforms and config files
            └── configs     <- YAML configs for different pipeline types and experiments
--------
*all experiments have been conducted using already pretrained models available on the HF Hub

## How to Run the Code from the Repository

### Creation of the Python Environment
To begin, you must create a virtual Python environment and install the necessary dependencies. Then, install all the modules listed in `requirements.txt` using the following command:
```bash
pip install -r requirements.txt
```

### Configuration File
All SLURM job scripts in the `slurm` folder take the path to a file containing the required environment variables as the first argument. This setup allows the user to run scripts in their own workspace, manage cache directories, configure MLFlow logging, etc. The configuration file should contain the following variables and resemble this example:
```plaintext
SRC_DIR=path_to_repo
WORKSPACE=path_to_workspace
VENV_DIR=path_to_python_virtual_env
MLFLOW_TRACKING_URI=link_to_the_mlflow_server
MLFLOW_TRACKING_PASSWORD=password # Escape special characters as necessary
MLFLOW_TRACKING_USERNAME=user_name
MLFLOW_EXPERIMENT_NAME=experiment_name
HF_HOME=huggingface_cache_dir
NLTK_DATA=nltk_cache_dir
```

## XLNER Pipelines

We have implemented various XLNER pipelines. To avoid configuration complexity (such as passing numerous parameters to CLI scripts), reuse almost all pipeline components (like translation, writing and reading arrow files, and sentence splitting), and simplify logging, we have introduced a general pipeline runner fully configurable via _YAML_ files. The configuration is logically split into two parts:
- **Pipeline**: Describes the pipeline itself and lists all pipeline steps.
- **Experiment**: Specifies parameters for the desired pipeline's steps. If some parameters are still unknown and need to be specified by the user, they are marked in the experiment config with the `???` symbol.

Unfortunately, used version of _hydra_ (the library we use for handling configs) introduces some limitations:
- You must specify the pipeline even if you've specified an experiment. However, you can determine which pipeline to specify by checking the experiment's defaults (though they are ignored by the old _hydra_).

You can find prepared pipeline and experiment configurations in the [src/pipelines/configs](src/pipelines/configs) directory.

### Structure of the Pipeline Config
The general structure of the pipeline config is straightforward, for example:
```yaml
pipeline:
  step1:
    deps: []
    transform:
      _target_: src.pipelines.TransformClass1
      argument_name: argument_value
  step2:
    deps: [step1]
    transform:
      _target_: src.pipelines.TransformClass2
      argument_name: argument_value
  step3:
    deps: [step1, step2]
    transform:
      _target_: src.pipelines.TransformClass3
      argument_name: argument_value
```

You should specify the pipeline as a list of steps. Each step has a name (e.g., `step1`) used to match dependencies (`deps`). A dependency means that the output of the specified step is an input to this step. You can specify multiple dependencies if the desired transform supports multiple inputs (inputs are provided to the transform in this case as a tuple).

The transform itself is specified in the `transform` field of the step and can be a class/function (`_target_` field) from any desired module, instantiated/called with the provided arguments. Available transforms can be found in the [src/pipelines/](src/pipelines/) directory, or you can implement your own.

### Pipeline Run Example
All predefined pipeline configs and experiments can be found in the `src/pipelines/configs` directory. More information about them is available in the README in this directory.

To run a pipeline, you must execute the following scripts:
```bash
python -m $SRC_DIR/src/pipelines/run_pipeline.py pipeline=<pipeline_name> experiment=<experiment_name> <expr_param1=param1value, ...>
```
In this command:
- `pipeline_name` refers to the file name (without the `.yaml` suffix) in the [src/pipelines/configs/pipeline](src/pipelines/configs/pipeline) directory.
- `<experiment_name>` refers to the file name in the [src/pipelines/configs/experiment](src/pipelines/configs/experiment) directory.

For example, to run a model transfer pipeline on the Estonian split of the XTREME dataset (you must specify the `WORKSPACE` environment variable):
```bash
python -m src.pipelines.run_pipeline \
  pipeline=model_transfer/eval_compare \
  experiment=model_transfer/xtreme \
  lang=et \
  ner_model=$NER_CAND_MODEL
```

If you want to use your own config instead of a predefined one, you must specify the path to the config directory and the config file name, for example:
```bash
python $SRC_DIR/src/pipelines/run_pipeline.py \
  --config-path <dir_with_configs> # Override default configs dir
  --config-name <your_desired_config_name>
```
or
```bash
python $SRC_DIR/src/pipelines/run_pipeline.py \
  --config-dir <dir_with_configs> # Extend default configs dir with your dir
  --config-name <your_desired_config_name>
```
For more information about working with configs, refer to the [official _hydra_ documentation](https://hydra.cc/docs/1.0/advanced/override_grammar/basic/).

Additionally, we have provided a script to simplify running the pipeline on SLURM:
```bash
sbatch $SRC_DIR/slurm/pipelines/run_pipeline.sh <path_to_config_file> <args_to_run_pipeline_py>
```
Please adjust slurm job's arguments to match requirements of your HPC system.

