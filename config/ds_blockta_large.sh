#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

config_json="$script_dir/config_blockta_large.json"
gpt_options=" \
       --block-lm \
       --task-mask \
       --bert-prob 0.5 \
       --gap-sentence-prob 0.3 \
       --avg-block-length 3 \
       --gpt-min-ratio 0.25 \
       --block-mask-prob 0.1 \
       --short-seq-prob 0.02 \
       --experiment-name blocklm-roberta-large-blank \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --seq-length 512 \
       --max-sequence-length 513 \
       --save /dataset/fd5061f6/english_data/checkpoints \
       --save-interval 100000 \
       --train-iters 500000 \
       --resume-dataloader \
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
       --lr-decay-iters 500000 \
       --lr-decay-ratio 0.025 \
       --warmup .06 \
       --checkpoint-activations \
       --fp16 \
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"