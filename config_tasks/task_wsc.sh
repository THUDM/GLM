TASK_NAME=wsc
EXPERIMENT_NAME=${MODEL_TYPE}-${TASK_NAME}
DATA_PATH="${DATA_ROOT}/WSC-negative"
MAX_SEQ_LEN=128

LR_RANGE=(1e-5)
EPOCH_RANGE=(20)

LR_SINGLE=1e-5
EPOCH_SINGLE=50

TRAIN_ARGS="--lr-decay-style linear \
            --warmup 0.1 \
            --weight-decay 0.1 \
            --loss-func mix \
            --wsc-negative \
            --length-penalty 1"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 1000 \
             --eval-iters 100"

PATTERN_IDS=(0 1 2)
PROMPT_IDS=(1 2 3)

BATCH_SIZE=16