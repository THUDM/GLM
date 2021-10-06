MODEL_TYPE="blank-large-sent"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --task-mask \
            --num-layers 24 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-position-embeddings 603 \
            --tokenizer-model-type bert-large-uncased \
            --tokenizer-type BertWordPieceTokenizer \
            --load-pretrained ${CHECKPOINT_PATH}/blocklm-large-sent-extend"