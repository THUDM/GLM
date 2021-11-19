#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

PRE_OPTIONS="--load /dataset/fd5061f6/english_data/checkpoints/blocklm-roberta-large-blank \
       --no-load-optim \
       --no-load-lr-scheduler \
       --no-load-rng \
       --no-load-iteration \
       --no-deepspeed-load \
       --old-checkpoint \
       --new-save-directory \
"

config_json="$script_dir/config_blockta_large.json"
gpt_options=" \
       --block-lm \
       --task-mask \
       --bert-prob 0.8 \
       --gap-sentence-prob 0.2 \
       --avg-block-length 3 \
       --gpt-min-ratio 0.25 \
       --short-seq-prob 0.02 \
       --experiment-name blocklm-roberta-large-blank \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --seq-length 512 \
       --max-sequence-length 513 \
       --prefix-prompt 100 \
       --save /dataset/fd5061f6/english_data/checkpoints \
       ${PRE_OPTIONS}
       --save-interval 5000 \
       --train-iters 50000 \
       --train-data wikibook cc-news openwebtext stories \
       --shuffle \
       --filter-english \
       --loader-scatter 8 \
       --no-lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --tokenizer-model-type roberta \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style linear \
       --lr-decay-iters 50000 \
       --lr-decay-ratio 0.1 \
       --warmup .1 \
       --checkpoint-activations \
       --fp16 \
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"