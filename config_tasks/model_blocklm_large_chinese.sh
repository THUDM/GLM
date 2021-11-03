MODEL_TYPE="blocklm-large-chinese"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --task-mask \
            --num-layers 24 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-position-embeddings 1024 \
            --tokenizer-type ChineseSPTokenizer \
            --load-pretrained ${CHECKPOINT_PATH}/blocklm-large-chinese"