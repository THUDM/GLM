MODEL_TYPE="blocklm-220M"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --task-mask \
            --num-layers 14 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-sequence-length 1025 \
            --tokenizer-type GPT2BPETokenizer \
            --tokenizer-model-type gpt2 \
            --old-checkpoint \
            --load-pretrained ${CHECKPOINT_PATH}/blocklm-220M08-08-16-09"