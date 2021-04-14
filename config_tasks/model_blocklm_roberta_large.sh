MODEL_TYPE="blocklm-roberta-large-250k"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --task-mask \
            --num-layers 24 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-position-embeddings 1024 \
            --tokenizer-model-type roberta \
            --tokenizer-type GPT2BPETokenizer \
            --load-pretrained /dataset/fd5061f6/english_data/checkpoints/blocklm-roberta-large-blank04-08-07-49"