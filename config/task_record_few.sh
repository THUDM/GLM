MODEL_ARGS="--block-lm \
            --cloze-eval \
            --num-layers 12 \
            --hidden-size 768 \
            --num-attention-heads 12 \
            --seq-length 512 \
            --max-position-embeddings 512 \
            --tokenizer-model-type bert-base-uncased \
            --tokenizer-type BertWordPieceTokenizer \
            --load-pretrained /root/data/checkpoints/block-lm-blank-cls12-18-12-50"

TRAIN_ARGS="--epochs 60 \
            --batch-size 8 \
            --lr 1e-5 \
            --lr-decay-style linear \
            --warmup 0.06 \
            --weight-decay 1.0e-1"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 10 \
             --eval-interval 1000 \
             --eval-epoch 10 \
             --eval-iters 100"