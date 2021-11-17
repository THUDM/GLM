MODEL_TYPE="blank-base"
MODEL_ARGS="--block-lm \
            --num-layers 12 \
            --hidden-size 768 \
            --num-attention-heads 12 \
            --max-sequence-length 513 \
            --tokenizer-model-type bert-base-uncased \
            --tokenizer-type BertWordPieceTokenizer \
            --old-checkpoint \
            --load-pretrained ${CHECKPOINT_PATH}/blocklm-base-blank"