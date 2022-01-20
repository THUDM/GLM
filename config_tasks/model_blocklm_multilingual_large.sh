MODEL_TYPE="blocklm-large-multilingual"
MODEL_ARGS="--block-lm \
            --task-mask \
            --cloze-eval \
            --num-layers 24 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-sequence-length 1025 \
            --tokenizer-type ChineseSPTokenizer \
            --tokenizer-model-type /dataset/fd5061f6/duzx16/tokenizer/mglm-unigram-250k/mglm250k-uni.model \
            --load-pretrained ${CHECKPOINT_PATH}/blocklm-large-multilingual"