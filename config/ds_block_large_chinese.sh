#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

config_json="$script_dir/config_block_large_chinese.json"
gpt_options=" \
       --block-lm \
       --task-mask \
       --bert-prob 0.4 \
       --gap-sentence-prob 0.3 \
       --avg-block-length 3 \
       --gpt-min-ratio 0.25 \
       --block-mask-prob 0.1 \
       --short-seq-prob 0.02 \
       --experiment-name blocklm-large-chinese \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --seq-length 512 \
       --max-sequence-length 1025 \
       --save /dataset/fd5061f6/english_data/checkpoints \
       --load /dataset/fd5061f6/english_data/checkpoints/blocklm-large-chinese08-12-12-48 \
       --log-interval 50 \
       --eval-interval 1000 \
       --save-interval 2000 \
       --train-iters 250000 \
       --train-data wudao baike zhihu \
       --resume-dataloader \
       --loader-scatter 32 \
       --no-lazy-loader \
       --tokenizer-type ChineseSPTokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --lr-decay-ratio 0.1 \
       --lr-decay-iters 200000 \
       --warmup 0.04 \
       --checkpoint-activations \
       --fp16 \
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"