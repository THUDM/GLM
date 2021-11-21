MODEL_TYPE="blank-large"
MODEL_ARGS="--block-lm \
            --num-layers 24 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-sequence-length 513 \
            --tokenizer-type BertWordPieceTokenizer \
            --tokenizer-model-type bert-large-uncased \
            --old-checkpoint \
            --load-pretrained ${CHECKPOINT_PATH}/blocklm-large-blank"