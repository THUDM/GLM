MODEL_TYPE="blocklm-1.5-generation"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --num-layers 30 \
            --hidden-size 1152 \
            --num-attention-heads 18 \
            --max-sequence-length 513 \
            --tokenizer-type BertWordPieceTokenizer \
            --tokenizer-model-type bert-large-uncased \
            --old-checkpoint \
            --load-pretrained ${CHECKPOINT_PATH}/blocklm-1.5-generation"