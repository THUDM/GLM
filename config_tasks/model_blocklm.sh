MODEL_TYPE="blank-base"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --num-layers 12 \
            --hidden-size 768 \
            --num-attention-heads 12 \
            --max-position-embeddings 512 \
            --tokenizer-model-type bert-base-uncased \
            --tokenizer-type BertWordPieceTokenizer \
            --load-pretrained /root/data/checkpoints/blocklm-base-generation"
