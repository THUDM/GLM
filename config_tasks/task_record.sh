EXPERIMENT_NAME=${MODEL_TYPE}-record
TASK_NAME=ReCoRD
DATA_PATH="${DATA_ROOT}/ReCoRD"
MAX_SEQ_LEN=448

LR_RANGE=(1e-5)
EPOCH_RANGE=(5)

LR_SINGLE=1e-5
EPOCH_SINGLE=1

TRAIN_ARGS="--lr-decay-style linear \
            --warmup 0.1 \
            --weight-decay 1.0e-1"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 1000 \
             --eval-iters 100"

PATTERN_IDS=(0)

BATCH_SIZE=64