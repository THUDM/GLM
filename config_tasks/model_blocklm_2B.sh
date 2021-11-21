MODEL_TYPE="blocklm-2B"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --task-mask \
            --num-layers 36 \
            --hidden-size 2048 \
            --num-attention-heads 32 \
            --max-position-embeddings 1024 \
            --tokenizer-type GPT2BPETokenizer \
            --tokenizer-model-type gpt2 \
            --old-checkpoint \
            --load-pretrained ${CHECKPOINT_PATH}/blocklm-xxlarge"