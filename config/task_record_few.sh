EXPERIMENT_NAME=blank-base-record-few-batch16
TASK_NAME=ReCoRD
DATA_PATH="/root/data/fewglue/ReCoRD"
MAX_SEQ_LEN=512
source config/model_blocklm.sh

TRAIN_ARGS="--epochs 100 \
            --batch-size 8 \
            --lr 1e-5 \
            --lr-decay-style linear \
            --warmup 0.06 \
            --weight-decay 1.0e-1
            --loss-func hinge"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 10 \
             --eval-interval 1000 \
             --eval-epoch 20 \
             --eval-iters 100"