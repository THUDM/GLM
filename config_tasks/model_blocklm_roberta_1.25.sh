MODEL_TYPE="blocklm-roberta-1.25"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --task-mask \
            --num-layers 24 \
            --hidden-size 1152 \
            --num-attention-heads 18 \
            --max-sequence-length 1025 \
            --attention-scale 8.0 \
            --tokenizer-type GPT2BPETokenizer \
            --tokenizer-model-type roberta \
            --old-checkpoint \
            --load-pretrained ${CHECKPOINT_PATH}/blocklm-roberta-1.25-blank04-22-14-01"