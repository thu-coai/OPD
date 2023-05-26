SHELL_SCRIPT_DIR="$( dirname "$0"  )"
PROJECT_DIR=$SHELL_SCRIPT_DIR/../..


INPUT_PATH=${PROJECT_DIR}/data/train.txt
OUTPUT_PATH=${PROJECT_DIR}/data/train

python tools/prepare_data.py \
    --input ${INPUT_PATH} \
    --output_path ${OUTPUT_PATH} \
    --tokenizer_path ${PROJECT_DIR}/vocab/vocab.txt \
    --max_seq_len 512