EXPERIMENT_NAME=${MODEL_TYPE}-mnli
TASK_NAME=mnli
DATA_PATH="${GLUE_DATA_ROOT}/MNLI"
MAX_SEQ_LEN=256

LR_SINGLE=1e-5
EPOCH_SINGLE=3

TRAIN_ARGS="--lr-decay-style linear \
            --warmup 0.1 \
            --weight-decay 1.0e-1"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 1000 \
             --eval-iters 100"

PATTERN_IDS=(0 1 2 3)
PROMPT_IDS=(1 2 3)

BATCH_SIZE=16