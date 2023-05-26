#! /bin/bash
PROJECT_DIR=''

set -ex


TEST_FILE=${PROJECT_DIR}/data/test.txt
OUT_FILE=${PROJECT_DIR}/data/gen.txt

CKPT=${PROJECT_DIR}/results/opd/checkpoint.pt

OPTS=""
OPTS+=" --model-config ${PROJECT_DIR}/src/config/opd.json"
OPTS+=" --vocab-file ${PROJECT_DIR}/vocab/vocab.txt"
OPTS+=" --load ${CKPT}"
OPTS+=" --input-file ${TEST_FILE}"
OPTS+=" --span-length 100"
OPTS+=" --temperature 0.9"
OPTS+=" --top-k 0"
OPTS+=" --top-p 0.9"
OPTS+=" --no-repeat-ngram-size 4"
OPTS+=" --repetition-penalty 1.2"
OPTS+=" --beam-size 4"
OPTS+=" --length-penalty 1.6"
OPTS+=" --random-sample" # 用于top_p, top_k sampling
OPTS+=" --output-file ${OUT_FILE}"
OPTS+=" --use_line_token_as_eos" # 以换行符'\n'作为生成结束标志

CMD="python3 ${PROJECT_DIR}/src/opd_inference_static.py ${OPTS}"
echo ${CMD}
${CMD}
set +ex
