#!/usr/bin/env bash
# Before running this set of experiments one should download
# the Europarl dataset into the $WORKSPACE/data/europarl directory
# using a script src/data/load_europarl_ds.py

CFG_FILE=$1

if [[ $# -eq 1 ]]
then
    NER_CAND_MODEL=FacebookAI/xlm-roberta-large-finetuned-conll03-english
elif [[ $# -eq 2 ]]
then
    NER_CAND_MODEL=$2
else
    echo "Illegal number of parameters" >&2
    exit 2
fi

source "$CFG_FILE"
export $(cut -d= -f1 "$CFG_FILE")

RUN="sbatch $SRC_DIR/slurm/pipelines/run_pipeline.sh $CFG_FILE"
DATA_DIR=$WORKSPACE/data/europarl
CACHE="+cached_step_outs={save_entities:'$DATA_DIR/en'}"
LABEL2ID="pipeline.save_generated_ds.transform.label2id_path='$WORKSPACE/pipelines/label2id_europarl.json'"

for lang in "de" "es" "it"
do
    LANG_TAG="mlflow_tags.lang=$lang"
    INPUT="input_args.open_orig='$DATA_DIR/$lang'"
    AWESOME_ALIGNER=awesome_en_$lang

    # Model transfer
    $RUN pipeline=model_transfer/eval_compare \
        log_to_mlflow=true \
        +mlflow_tags.dataset=europarl \
        ner_model=$NER_CAND_MODEL \
        $INPUT $LANG_TAG $LABEL2ID

    # Target candidates based
    $RUN pipeline=candidates/dummy_argmax_eval_compare \
            experiment=europarl \
            word_aligner=simalign_mbert_iterative \
            use_cached=True $LANG_TAG $INPUT $CACHE $LABEL2ID

    $RUN pipeline=candidates/ner_argmax_eval_compare \
            experiment=europarl \
            ner_cand_model=$NER_CAND_MODEL \
            word_aligner=simalign_mbert_iterative \
            use_cached=True $LANG_TAG $INPUT $CACHE $LABEL2ID

    $RUN pipeline=candidates/dummy_argmax_eval_compare \
            experiment=europarl \
            word_aligner=$AWESOME_ALIGNER \
            use_cached=True $LANG_TAG $INPUT $CACHE $LABEL2ID

    $RUN pipeline=candidates/ner_argmax_eval_compare \
            experiment=europarl \
            ner_cand_model=$NER_CAND_MODEL \
            word_aligner=$AWESOME_ALIGNER \
            use_cached=True $LANG_TAG $INPUT $CACHE $LABEL2ID

    # Word to word alignments based
    $RUN pipeline=align/src2tgt_eval_compare \
            experiment=europarl \
            word_aligner=simalign_mbert_iterative \
            use_cached=True $LANG_TAG $INPUT $CACHE $LABEL2ID

    $RUN pipeline=align/src2tgt_eval_compare \
            experiment=europarl \
            word_aligner=$AWESOME_ALIGNER \
            use_cached=True $LANG_TAG $INPUT $CACHE $LABEL2ID

done
