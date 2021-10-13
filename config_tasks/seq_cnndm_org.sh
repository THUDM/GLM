EXPERIMENT_NAME=${MODEL_TYPE}-cnndm_org
TASK_NAME=cnn_dm_original
DATA_PATH="${DATA_ROOT}/cnn_dm_original"

TRAIN_ARGS="--epochs 20 \
            --batch-size 16 \
            --lr 1e-5 \
            --lr-decay-style linear \
            --warmup 0.06 \
            --weight-decay 1.0e-1 \
            --label-smoothing 0.1"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 1000 \
             --eval-iters 100 \
             --eval-epoch 2 \
             --save-epoch 2"

TASK_ARGS="--src-seq-length 608 \
           --tgt-seq-length 160 \
           --min-tgt-length 55 \
           --length-penalty 0.7 \
           --no-repeat-ngram-size 3 \
           --num-beams 5 \
           --select-topk \
           --eval-batch-size 1"