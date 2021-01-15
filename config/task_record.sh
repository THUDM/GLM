EXPERIMENT_NAME=blank-base-record
TASK_NAME=ReCoRD
DATA_PATH="/root/data/superglue/ReCoRD"
MAX_SEQ_LEN=512
source config/task_blocklm.sh

TRAIN_ARGS="--epochs 5 \
            --batch-size 8 \
            --lr 1e-5 \
            --lr-decay-style linear \
            --warmup 0.06 \
            --weight-decay 1.0e-1"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 100 \
             --eval-interval 1000 \
             --eval-iters 100"