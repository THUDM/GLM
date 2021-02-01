source config/model_blocklm_generation.sh
EXPERIMENT_NAME=${MODEL_TYPE}-record
TASK_NAME=ReCoRD
DATA_PATH="/root/data/superglue/ReCoRD"
MAX_SEQ_LEN=512

TRAIN_ARGS="--epochs 4 \
            --batch-size 4 \
            --lr 1e-5 \
            --lr-decay-style linear \
            --warmup 0.06 \
            --weight-decay 1.0e-1"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 1000 \
             --eval-interval 1000 \
             --eval-iters 100"