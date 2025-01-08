CFG_FILE=$1

if [[ $# -eq 1 ]]
then
    SRC_NER_MODEL=FacebookAI/xlm-roberta-large-finetuned-conll03-english
    NER_CAND_MODEL=$SRC_NER_MODEL
elif [[ $# -eq 2 ]]
then
    SRC_NER_MODEL=$2
    NER_CAND_MODEL=$SRC_NER_MODEL
elif [[ $# -eq 3 ]]
then
    SRC_NER_MODEL=$2
    NER_CAND_MODEL=$3
else
    echo "Illegal number of parameters" >&2
    exit 2
fi

source "$CFG_FILE"
export $(cut -d= -f1 "$CFG_FILE")

langs=("af" "ar" "bg" "bn" "de" "el" "es" "et" "eu" "fa" "fi" "fr" "he" "hi" "hu" "id" "it"
        "ja" "jv" "ka" "kk" "ko" "ml" "mr" "ms" "my" "nl" "pt" "ru" "sw" "ta" "te" "th" "tl"
        "tr" "ur" "vi" "yo" "zh")
lang_codes=("afr_Latn" "arb_Arab" "bul_Cyrl" "ben_Beng" "deu_Latn" "ell_Grek" "spa_Latn" "est_Latn"
    "eus_Latn" "pes_Arab" "fin_Latn" "fra_Latn" "heb_Hebr" "hin_Deva" "hun_Latn" "ind_Latn"
    "ita_Latn" "jpn_Jpan" "jav_Latn" "kat_Geor" "kaz_Cyrl" "kor_Hang" "mal_Mlym" "mar_Deva"
    "zsm_Latn" "mya_Mymr" "nld_Latn" "por_Latn" "rus_Cyrl" "swh_Latn" "tam_Taml" "tel_Telu"
    "tha_Thai" "tgl_Latn" "tur_Latn" "urd_Arab" "vie_Latn" "yor_Latn" "zho_Hans")

for i in "${!langs[@]}"; do
    lang="${langs[i]}"
    lang_code="${lang_codes[i]}"

    sbatch $SRC_DIR/slurm/pipelines/xtreme/run_lang.sh $CFG_FILE $lang $lang_code $SRC_NER_MODEL $NER_CAND_MODEL
done
