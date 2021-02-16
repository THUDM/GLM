MODEL_TYPE="blank-large-sentinel"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --sentinel-token \
            --num-layers 24 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-position-embeddings 512 \
            --tokenizer-model-type bert-large-uncased \
            --tokenizer-type BertWordPieceTokenizer \
            --load-pretrained /root/data/checkpoints/blocklm-large-blank-sentinel02-15-12-57"