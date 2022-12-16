#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

config_json="$script_dir/config_block_large.json"
gpt_options=" \
      --block-lm \
      --task-mask \
      --bert-prob 0.5 \
      --gap-sentence-prob 0.3 \
      --avg-block-length 3 \
      --gpt-min-ratio 0.25 \
      --experiment-name blocklm-large-blank \
      --model-parallel-size ${MP_SIZE} \
      --num-layers 12 \
      --hidden-size 512 \
      --num-attention-heads 8 \
      --seq-length 256 \
      --max-position-embeddings 256 \
      --save /root/data/checkpoints \
      --train-iters 1000 \
      --resume-dataloader \
      --train-data bert-large \
      --tokenizer-type BertWordPieceTokenizer \
      --tokenizer-model-type bert-large-uncased \
      --split 949,50,1 \
      --distributed-backend nccl \
      --lr-decay-style cosine \
      --lr-decay-iters 160000 \
      --lr-decay-ratio 0.05 \
      --warmup .05 \
"
# gpt_options="${gpt_options}
#                --deepspeed \
#                --deepspeed_config ${config_json} \
# "
