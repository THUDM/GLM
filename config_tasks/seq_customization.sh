EXPERIMENT_NAME=${MODEL_TYPE}-customization
TASK_NAME=customization
DATA_PATH="${DATA_ROOT}/customization"

TRAIN_ARGS="--epochs 10 \
            --lr 1e-5 \
            --lr-decay-style linear \
            --warmup 0.06 \
            --label-smoothing 0.1"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 1000 \
             --eval-iters 100 \
             --eval-epoch 2"

TASK_ARGS="--src-seq-length 512 \
           --tgt-seq-length 128 \
           --min-tgt-length 55 \
           --length-penalty 0.7 \
           --no-repeat-ngram-size 3 \
           --num-beams 5 \
           --select-topk \
           --eval-batch-size 1"