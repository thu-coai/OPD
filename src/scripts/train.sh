#! /bin/bash

set -ex

SHELL_SCRIPT_DIR="$( dirname "$0"  )"
PROJECT_DIR=$SHELL_SCRIPT_DIR/../..

MASTER_ADDR=127.0.0.1
MASTER_PORT=6566
NNODES=1
NODE_RANK=0

GPUS_PER_NODE=4

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

TRAIN_BATCH=1
LR=1e-5
WARM_UP=100


CKPT_PATH=${PROJECT_DIR}/results/opd/checkpoint.pt
SAVE_DIR=${PROJECT_DIR}/results/opd-finetune

DATASET=${PROJECT_DIR}/data
TUNING=fine


OPTS=""
OPTS+=" --model-config ${PROJECT_DIR}/src/config/opd.json"
OPTS+=" --vocab-file ${PROJECT_DIR}/vocab/vocab.txt"
OPTS+=" --batch-size ${TRAIN_BATCH}"
OPTS+=" --train-iters 50000"
OPTS+=" --save-iters 500"
OPTS+=" --save-name tuning-checkpoint"
OPTS+=" --max-length 576"
OPTS+=" --save $SAVE_DIR"
OPTS+=" --lr ${LR}"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters ${WARM_UP}"
OPTS+=" --lr-decay-style linear"
OPTS+=" --weight-decay 0.01"
OPTS+=" --clip-grad 4.0"
OPTS+=" --loss-scale 1048576"
OPTS+=" --start-step 0"
OPTS+=" --load ${CKPT_PATH}"
OPTS+=" --eval-step 1000"
OPTS+=" --eval-batch-size 64"
OPTS+=" --dataset pku"
OPTS+=" --task rewrite"
OPTS+=" --log-dir ${SAVE_DIR}"
OPTS+=" --epochs 10"
OPTS+=" --dataset_main_path ${DATASET}"

if [[ $TUNING == prompt ]]; then
    OPTS+=" --prompt-tuning"
fi

export PYTHONIOENCODING=utf-8
CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${PROJECT_DIR}/src/opd_train.py ${OPTS}"
if [ ! -d "$SAVE_DIR" ]; then
  mkdir -p "$SAVE_DIR"
fi

echo ${CMD} > ${SAVE_DIR}/cmd.txt

if [[ $NODE_RANK == 0 ]]; then
    ${CMD} 2>&1 | tee ${SAVE_DIR}/log
else
    ${CMD}
fi