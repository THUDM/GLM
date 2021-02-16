MODEL_TYPE="blank-large-no2d"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --no-block-position \
            --num-layers 24 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-position-embeddings 512 \
            --tokenizer-model-type bert-large-uncased \
            --tokenizer-type BertWordPieceTokenizer \
            --load-pretrained /root/data/checkpoints/blocklm-large-blank-no2d02-15-12-36"