MODEL_TYPE="blocklm-roberta-1.25"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --task-mask \
            --num-layers 24 \
            --hidden-size 1152 \
            --num-attention-heads 18 \
            --max-position-embeddings 1024 \
            --attention-scale 8.0 \
            --tokenizer-model-type roberta \
            --tokenizer-type GPT2BPETokenizer \
            --load-pretrained ${CHECKPOINT_PATH}/blocklm-roberta-1.25-blank04-22-14-01"