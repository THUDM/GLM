MODEL_TYPE="blocklm-220M"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --task-mask \
            --num-layers 14 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-position-embeddings 1024 \
            --tokenizer-type GPT2BPETokenizer \
            --load-pretrained ${CHECKPOINT_PATH}/blocklm-220M08-08-16-09"