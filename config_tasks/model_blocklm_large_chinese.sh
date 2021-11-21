MODEL_TYPE="blocklm-large-chinese"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --task-mask \
            --num-layers 24 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-sequence-length 1025 \
            --tokenizer-type ChineseSPTokenizer \
            --tokenizer-model-type glm-large \
            --old-checkpoint \
            --load-pretrained ${CHECKPOINT_PATH}/blocklm-large-chinese"