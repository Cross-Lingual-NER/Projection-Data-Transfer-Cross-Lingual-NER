#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --account=account_name
#SBATCH --job-name=xtreme_pipelines
#SBATCH --gres=gpu:1
#SBATCH --partition=partition_name
#SBATCH --mem=64G
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
lang_code=$3
SRC_NER_MODEL=$4
NER_CAND_MODEL=$5

if [[ $lang == @("ja"|"th") ]]; then
    back_trans_splitter_tags="pipeline.back_translation.transform.word_splitter._target_=src.pipelines.word_splitting.JapaneseThaiSplitter"
    detok_tags="+pipeline.detokenize.transform.lang_has_whitespace=False"
    ngram_limit="+pipeline.candidate_extraction.transform.extractor.max_words=30"
elif [[ $lang == "zh" ]]; then
    back_trans_splitter_tags="pipeline.back_translation.transform.word_splitter._target_=src.pipelines.word_splitting.ChineseSplitter"
    detok_tags="+pipeline.detokenize.transform.lang_has_whitespace=False"
    ngram_limit="+pipeline.candidate_extraction.transform.extractor.max_words=30"
else
    back_trans_splitter_tags=''
    detok_tags=''
    ngram_limit=''
fi

RUN="python -m src.pipelines.run_pipeline"
trans_lang_code="pipeline.fwd_translation.transform.src_lang_code="$lang_code

echo "Start experiments for the $lang language"

# Cache translations and NER labaling for all pipeline's types
PIPE_CACHE_DIR=$WORKSPACE/data/xtreme40/cache/$lang
mkdir -p $PIPE_CACHE_DIR

FWD_TRANS_PATH=$PIPE_CACHE_DIR/fwd_translation_nllb_${lang}_eng.arrow
if [ -f $FWD_TRANS_PATH ]; then
   echo "Use cached translation"
else
    echo "[PIPELINE] Start forward translation"
    $RUN pipeline=nllb_fwd_trans \
        src_lang=$lang tgt_lang=eng \
        pipeline.fwd_translation.transform.src_lang_code=$lang_code \
        input_args.open_orig='google/xtreme' \
        pipeline.open_orig.transform.cfg_name=PAN-X.$lang \
        pipeline.save_translation.transform.path=$FWD_TRANS_PATH \
        $detok_tags

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
        src_lang=eng tgt_lang=$lang \
        pipeline.back_translation.transform.tgt_lang_code=$lang_code \
        input_args.open_src_entities=$SRC_ENTITIES_PATH \
        pipeline.save_entities.transform.path=$TGT_ENTITIES_PATH \
        $back_trans_splitter_tags
fi

# Model transfer
echo "[PIPELINE] Start model transfer pipeline"
$RUN pipeline=model_transfer/eval_compare \
    experiment=model_transfer/xtreme \
    lang=$lang \
    ner_model=$NER_CAND_MODEL

simalign_tag="+mlflow_tags.aligner=simalign"
awesome_tag="+mlflow_tags.aligner=awesome"


# Target candidates based
## Src2Tgt
echo "[PIPELINE] Start src2tgt ngram candidates pipeline with simalign"
$RUN pipeline=candidates/dummy_argmax_eval_compare \
        experiment=candidates/dummy_argmax_xtreme \
        word_aligner=simalign_mbert_iterative $simalign_tag \
        lang=$lang $trans_lang_code $ngram_limit \
        use_cached=true \
        +cached_step_outs={save_entities:$SRC_ENTITIES_PATH}

echo "[PIPELINE] Start src2tgt NER candidates pipeline with simalign"
$RUN pipeline=candidates/ner_argmax_eval_compare \
        experiment=candidates/ner_argmax_xtreme \
        ner_cand_model=$NER_CAND_MODEL \
        word_aligner=simalign_mbert_iterative $simalign_tag \
        lang=$lang $trans_lang_code \
        use_cached=true \
        +cached_step_outs={save_entities:$SRC_ENTITIES_PATH}

echo "[PIPELINE] Start src2tgt ngram candidates pipeline with awesome"
$RUN pipeline=candidates/dummy_argmax_eval_compare \
        experiment=candidates/dummy_argmax_xtreme \
        word_aligner=awesome_mbert $awesome_tag \
        lang=$lang $trans_lang_code $ngram_limit \
        use_cached=true \
        +cached_step_outs={save_entities:$SRC_ENTITIES_PATH}

echo "[PIPELINE] Start src2tgt NER candidates pipeline with awesome"
$RUN pipeline=candidates/ner_argmax_eval_compare \
        experiment=candidates/ner_argmax_xtreme \
        ner_cand_model=$NER_CAND_MODEL \
        word_aligner=awesome_mbert $awesome_tag \
        lang=$lang $trans_lang_code \
        use_cached=true \
        +cached_step_outs={save_entities:$SRC_ENTITIES_PATH}

## Tgt2Tgt
echo "[PIPELINE] Start tgt2tgt ngram candidates pipeline with simalign"
$RUN pipeline=candidates/backtrans_dummy_argmax_eval_compare \
        experiment=candidates/backtrans_dummy_argmax_xtreme \
        word_aligner=simalign_mbert_iterative $simalign_tag \
        lang=$lang $trans_lang_code $ngram_limit \
        use_cached=true \
        +cached_step_outs={save_entities:$TGT_ENTITIES_PATH}

echo "[PIPELINE] Start tgt2tgt NER candidates pipeline with simalign"
$RUN pipeline=candidates/backtrans_ner_argmax_eval_compare \
        experiment=candidates/backtrans_ner_argmax_xtreme \
        ner_cand_model=$NER_CAND_MODEL \
        word_aligner=simalign_mbert_iterative $simalign_tag \
        lang=$lang $trans_lang_code \
        use_cached=true \
        +cached_step_outs={save_entities:$TGT_ENTITIES_PATH}

echo "[PIPELINE] Start tgt2tgt ngram candidates pipeline with awesome"
$RUN pipeline=candidates/backtrans_dummy_argmax_eval_compare \
        experiment=candidates/backtrans_dummy_argmax_xtreme \
        word_aligner=awesome_mbert $awesome_tag \
        lang=$lang $trans_lang_code $ngram_limit \
        use_cached=true \
        +cached_step_outs={save_entities:$TGT_ENTITIES_PATH}

echo "[PIPELINE] Start tgt2tgt NER candidates pipeline with awesome"
$RUN pipeline=candidates/backtrans_ner_argmax_eval_compare \
        experiment=candidates/backtrans_ner_argmax_xtreme \
        ner_cand_model=$NER_CAND_MODEL \
        word_aligner=awesome_mbert $awesome_tag \
        lang=$lang $trans_lang_code \
        use_cached=true \
        +cached_step_outs={save_entities:$TGT_ENTITIES_PATH}

# Word to word alignments based
## Src2Tgt
echo "[PIPELINE] Start src2tgt align pipeline with simalign"
$RUN pipeline=align/src2tgt_eval_compare \
        experiment=align/src2tgt_xtreme \
        word_aligner=simalign_mbert_iterative $simalign_tag \
        lang=$lang $trans_lang_code \
        use_cached=true \
        +cached_step_outs={save_entities:$SRC_ENTITIES_PATH}

echo "[PIPELINE] Start src2tgt align pipeline with awesome"
$RUN pipeline=align/src2tgt_eval_compare \
        experiment=align/src2tgt_xtreme \
        word_aligner=awesome_mbert $awesome_tag \
        lang=$lang $trans_lang_code \
        use_cached=true \
        +cached_step_outs={save_entities:$SRC_ENTITIES_PATH}

## Tgt2Tgt
echo "[PIPELINE] Start tgt2tgt align pipeline with simalign"
$RUN pipeline=align/tgt2tgt_eval_compare \
        experiment=align/tgt2tgt_xtreme \
        word_aligner=simalign_mbert_iterative $simalign_tag \
        lang=$lang $trans_lang_code \
        use_cached=true \
        +cached_step_outs={save_entities:$TGT_ENTITIES_PATH}

echo "[PIPELINE] Start tgt2tgt align pipeline with awesome"
$RUN pipeline=align/tgt2tgt_eval_compare \
        experiment=align/tgt2tgt_xtreme \
        word_aligner=awesome_mbert $awesome_tag \
        lang=$lang $trans_lang_code \
        use_cached=true \
        +cached_step_outs={save_entities:$TGT_ENTITIES_PATH}