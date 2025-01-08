#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --account=account_name
#SBATCH --job-name=masakhaner2_pipelines
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

lang=$2
SRC_NER_MODEL=$3
NER_CAND_MODEL=$4

RUN="python -m src.pipelines.run_pipeline"

if [ "$lang" = "swa" ]
then
    trans_lang_code="pipeline.fwd_translation.transform.src_lang_code=swh_Latn"
    backtrans_lang_code="pipeline.back_translation.transform.tgt_lang_code=swh_Latn"
else
    trans_lang_code=''
    backtrans_lang_code=''
fi

# Cache translations and NER labaling for all pipeline's types
PIPE_CACHE_DIR=$WORKSPACE/data/masakhaner2/cache/$lang
mkdir -p $PIPE_CACHE_DIR

FWD_TRANS_PATH=$PIPE_CACHE_DIR/fwd_translation_nllb_${lang}_eng.arrow
if [ -f $FWD_TRANS_PATH ]; then
   echo "Use cached translation"
else
    echo "[PIPELINE] Start forward translation"
    $RUN pipeline=nllb_fwd_trans \
        src_lang=$lang tgt_lang=eng $trans_lang_code \
        input_args.open_orig='masakhane/masakhaner2' \
        pipeline.open_orig.transform.cfg_name=$lang \
        pipeline.save_translation.transform.path=$FWD_TRANS_PATH

    if [ $? -ne 0 ]; then
      echo "Error during forward translation!"
      exit 1
    fi
fi

ner_model_name=$(echo $SRC_NER_MODEL | sed  's\/\_\g')
SRC_ENTITIES_PATH=$PIPE_CACHE_DIR/src_entities_$ner_model_name.arrow
if [ -f $SRC_ENTITIES_PATH ]; then
   echo "Use cached source entities"
else
    echo "[PIPELINE] Start SRC NER labeling"
    $RUN pipeline=src_ner \
        src_lang=eng ner_model=$SRC_NER_MODEL \
        input_args.open_translation=$FWD_TRANS_PATH \
        pipeline.save_entities.transform.path=$SRC_ENTITIES_PATH

    if [ $? -ne 0 ]; then
      echo "Error during source NER labeling!"
      exit 1
    fi
fi

TGT_ENTITIES_PATH=$PIPE_CACHE_DIR/tgt_entities_$ner_model_name.arrow
if [ -f $TGT_ENTITIES_PATH ]; then
   echo "Use cached backtranslated entities"
else
    echo "[PIPELINE] Start backtranslation"
    $RUN pipeline=easy_project_backtrans \
        src_lang=eng tgt_lang=$lang $backtrans_lang_code \
        input_args.open_src_entities=$SRC_ENTITIES_PATH \
        pipeline.save_entities.transform.path=$TGT_ENTITIES_PATH
fi

# Model transfer
echo "[PIPELINE] Start model transfer pipeline"
$RUN pipeline=model_transfer/eval_compare \
    experiment=model_transfer/masakhaner2 \
    lang=$lang \
    ner_model=$NER_CAND_MODEL

simalign_tag="+mlflow_tags.aligner=simalign"
awesome_tag="+mlflow_tags.aligner=awesome"

# Target candidates based
## Src2Tgt
echo "[PIPELINE] Start src2tgt ngram candidates pipeline with simalign"
$RUN pipeline=candidates/dummy_argmax_eval_compare \
        experiment=candidates/dummy_argmax_masakha_nllb \
        word_aligner=simalign_mbert_iterative $simalign_tag \
        lang=$lang $trans_lang_code \
        use_cached=true \
        +cached_step_outs={save_entities:$SRC_ENTITIES_PATH}

echo "[PIPELINE] Start src2tgt NER candidates pipeline with simalign"
$RUN pipeline=candidates/ner_argmax_eval_compare \
        experiment=candidates/ner_argmax_masakha_nllb \
        ner_cand_model=$NER_CAND_MODEL \
        word_aligner=simalign_mbert_iterative $simalign_tag \
        lang=$lang $trans_lang_code \
        use_cached=true \
        +cached_step_outs={save_entities:$SRC_ENTITIES_PATH}

echo "[PIPELINE] Start src2tgt ngram candidates pipeline with awesome"
$RUN pipeline=candidates/dummy_argmax_eval_compare \
        experiment=candidates/dummy_argmax_masakha_nllb \
        word_aligner=awesome_mbert  $awesome_tag\
        lang=$lang $trans_lang_code \
        use_cached=true \
        +cached_step_outs={save_entities:$SRC_ENTITIES_PATH}

echo "[PIPELINE] Start src2tgt NER candidates pipeline with awesome"
$RUN pipeline=candidates/ner_argmax_eval_compare \
        experiment=candidates/ner_argmax_masakha_nllb \
        ner_cand_model=$NER_CAND_MODEL \
        word_aligner=awesome_mbert  $awesome_tag\
        lang=$lang $trans_lang_code \
        use_cached=true \
        +cached_step_outs={save_entities:$SRC_ENTITIES_PATH}

## Tgt2Tgt
echo "[PIPELINE] Start tgt2tgt ngram candidates pipeline with sinalign"
$RUN pipeline=candidates/backtrans_dummy_argmax_eval_compare \
        experiment=candidates/backtrans_dummy_argmax_masakha_nllb \
        word_aligner=simalign_mbert_iterative $simalign_tag \
        lang=$lang $trans_lang_code \
        use_cached=true \
        +cached_step_outs={save_entities:$TGT_ENTITIES_PATH}

echo "[PIPELINE] Start tgt2tgt NER candidates pipeline with simalign"
$RUN pipeline=candidates/backtrans_ner_argmax_eval_compare \
        experiment=candidates/backtrans_ner_argmax_masakha_nllb \
        ner_cand_model=$NER_CAND_MODEL \
        word_aligner=simalign_mbert_iterative $simalign_tag \
        lang=$lang $trans_lang_code \
        use_cached=true \
        +cached_step_outs={save_entities:$TGT_ENTITIES_PATH}

echo "[PIPELINE] Start tgt2tgt ngram candidates pipeline with awesome"
$RUN pipeline=candidates/backtrans_dummy_argmax_eval_compare \
        experiment=candidates/backtrans_dummy_argmax_masakha_nllb \
        word_aligner=awesome_mbert  $awesome_tag\
        lang=$lang $trans_lang_code \
        use_cached=true \
        +cached_step_outs={save_entities:$TGT_ENTITIES_PATH}

echo "[PIPELINE] Start tgt2tgt NER candidates pipeline with awesome"
$RUN pipeline=candidates/backtrans_ner_argmax_eval_compare \
        experiment=candidates/backtrans_ner_argmax_masakha_nllb \
        ner_cand_model=$NER_CAND_MODEL \
        word_aligner=awesome_mbert  $awesome_tag\
        lang=$lang $trans_lang_code \
        use_cached=true \
        +cached_step_outs={save_entities:$TGT_ENTITIES_PATH}

# Word to word alignments based
## Src2Tgt
echo "[PIPELINE] Start src2tgt align pipeline with simalign"
$RUN pipeline=align/src2tgt_eval_compare \
        experiment=align/src2tgt_masakhaner_nllb \
        word_aligner=simalign_mbert_iterative $simalign_tag \
        lang=$lang $trans_lang_code \
        use_cached=true \
        +cached_step_outs={save_entities:$SRC_ENTITIES_PATH}

echo "[PIPELINE] Start src2tgt align pipeline with awesome"
$RUN pipeline=align/src2tgt_eval_compare \
        experiment=align/src2tgt_masakhaner_nllb \
        word_aligner=awesome_mbert  $awesome_tag\
        lang=$lang $trans_lang_code \
        use_cached=true \
        +cached_step_outs={save_entities:$SRC_ENTITIES_PATH}

## Tgt2Tgt
echo "[PIPELINE] Start tgt2tgt align pipeline with simalign"
$RUN pipeline=align/tgt2tgt_eval_compare \
        experiment=align/tgt2tgt_masakhaner_nllb \
        word_aligner=simalign_mbert_iterative $simalign_tag \
        lang=$lang $trans_lang_code \
        use_cached=true \
        +cached_step_outs={save_entities:$TGT_ENTITIES_PATH}

echo "[PIPELINE] Start tgt2tgt align pipeline with awesome"
$RUN pipeline=align/tgt2tgt_eval_compare \
        experiment=align/tgt2tgt_masakhaner_nllb \
        word_aligner=awesome_mbert  $awesome_tag\
        lang=$lang $trans_lang_code \
        use_cached=true \
        +cached_step_outs={save_entities:$TGT_ENTITIES_PATH}