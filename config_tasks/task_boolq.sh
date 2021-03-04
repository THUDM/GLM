EXPERIMENT_NAME=${MODEL_TYPE}-boolq
TASK_NAME=BoolQ
DATA_PATH="/root/data/superglue/BoolQ"
MAX_SEQ_LEN=256

LR_RANGE=(1e-5)
EPOCH_RANGE=(10)

LR_SINGLE=1e-5
EPOCH_SINGLE=10

TRAIN_ARGS="--lr-decay-style linear \
            --warmup 0.1 \
            --weight-decay 1.0e-1"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 1000 \
             --eval-iters 100"