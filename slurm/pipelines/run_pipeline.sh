#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --account=account_name
#SBATCH --job-name=pipeline
#SBATCH --gres=gpu:1
#SBATCH --partition=partition_name
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --mail-type=end,fail

module switch release/23.04
module load GCCcore/11.3.0
module load Python/3.10.4
module load CUDA/11.8.0

nvidia-smi

CFG_FILE=$1

source "$CFG_FILE"
export $(cut -d= -f1 "$CFG_FILE")

source $VENV_DIR/bin/activate

python -m src.pipelines.run_pipeline "${@:2}"
