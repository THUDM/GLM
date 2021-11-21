MODEL_TYPE="blank-base"
MODEL_ARGS="--block-lm \
            --num-layers 12 \
            --hidden-size 768 \
            --num-attention-heads 12 \
            --max-sequence-length 513 \
            --tokenizer-type BertWordPieceTokenizer \
            --tokenizer-model-type bert-base-uncased \
            --old-checkpoint \
            --load-pretrained ${CHECKPOINT_PATH}/blocklm-base-blank"