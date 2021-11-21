MODEL_TYPE="blocklm-roberta-large-pre"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --num-layers 24 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-sequence-length 513 \
            --tokenizer-type GPT2BPETokenizer \
            --tokenizer-model-type roberta \
            --load-pretrained ${CHECKPOINT_PATH}/blocklm-roberta-large-blank11-08-09-21"