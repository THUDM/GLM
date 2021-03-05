MODEL_TYPE="blocklm-roberta-large-250k"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --num-layers 24 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-position-embeddings 512 \
            --tokenizer-model-type roberta \
            --tokenizer-type GPT2BPETokenizer \
            --load-pretrained /root/data/checkpoints/blocklm-roberta-large/250000"