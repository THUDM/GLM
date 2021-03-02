EXPERIMENT_NAME=${MODEL_TYPE}-WSC
TASK_NAME=wsc
DATA_PATH="/root/data/superglue/WSC-negative"
MAX_SEQ_LEN=128

LR_RANGE=(1e-5)
EPOCH_RANGE=(50)

LR_SINGLE=1e-5
EPOCH_SINGLE=50

TRAIN_ARGS="--batch-size 8 \
            --lr-decay-style linear \
            --warmup 0.1 \
            --weight-decay 0.1 \
            --wsc-negative \
            --length-penalty 1 \
            --gradient-accumulation-steps 4"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 1000 \
             --eval-iters 100"