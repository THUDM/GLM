MODEL_TYPE="blank-large-sent"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --task-mask \
            --num-layers 24 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-sequence-length 604 \
            --tokenizer-type BertWordPieceTokenizer \
            --tokenizer-model-type bert-large-uncased \
            --old-checkpoint \
            --load-pretrained ${CHECKPOINT_PATH}/blocklm-large-sent-extend"