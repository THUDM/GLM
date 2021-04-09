EXPERIMENT_NAME=${MODEL_TYPE}-blank-$1
TASK_NAME=blank
DATA_PATH="/root/data/blank_yahoo"

TRAIN_ARGS="--epochs 5 \
            --batch-size 16 \
            --lr 1e-5 \
            --lr-decay-style linear \
            --warmup 0.06 \
            --weight-decay 1.0e-1
            --label-smoothing 0.1 \
            --blank-maskratio $1 \
            --save-epoch 5"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 1000 \
             --eval-iters 100"

TASK_ARGS="--src-seq-length 256 \
           --tgt-seq-length 200 \
           --min-tgt-length 0 \
           --length-penalty 1 \
           --no-repeat-ngram-size 3 \
           --eval-batch-size 8"