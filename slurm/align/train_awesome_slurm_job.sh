#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --account=account_name
#SBATCH --job-name=train_awesome_align
#SBATCH --gres=gpu:2
#SBATCH --partition=partition_name
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
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

awesome-train "${@:2}"

# Example args
# --model_name_or_path=bert-base-multilingual-cased \
# --extraction 'softmax' \
# --output_dir=awesome_en_es \
# --do_train \
# --train_mlm     --train_tlm     --train_tlm_full     --train_so     --train_psi \
# --train_data_file=$(realpath ../data/awesome/en_es/align.en-es) \
# --per_gpu_train_batch_size 16     --num_train_epochs 1     --learning_rate 2e-5 \
# --save_steps 10000 \
# --max_steps 40000
