MODEL_TYPE="blocklm-10B-chinese"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --task-mask \
            --num-layers 48 \
            --hidden-size 4096 \
            --num-attention-heads 64 \
            --max-sequence-length 1025 \
            --tokenizer-type ChineseSPTokenizer \
            --tokenizer-model-type glm-10b \
            --no-fix-command \
            --load-pretrained ${CHECKPOINT_PATH}/blocklm-10b-chinese07-08-15-28"