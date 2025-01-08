#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --account=account_name
#SBATCH --job-name=candidate_ner_train
#SBATCH --gres=gpu:1
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

python -m src.models.ner.candidates.train "${@:2}"

# Example args
# --model_name_or_path FacebookAI/xlm-roberta-large \
# --dataset_name eriktks/conll2003 \
# --output_dir conll2003/xlm_roberta_large_10epoch \
# --return_entity_level_metrics \
# --do_train --fp16 --num_train_epochs 10 --gradient_accumulation_steps 4 \
# --lambda_loss 3 \
# --auto_find_batch_size \
# --do_eval --do_predict \
# --save_strategy epoch \
# --overwrite_output_dir
