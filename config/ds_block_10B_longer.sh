#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

config_json="$script_dir/config_block_10B_longer.json"
gpt_options=" \
       --block-lm \
       --task-mask \
       --bert-prob 0.4 \
       --gap-sentence-prob 0.3 \
       --single-span-prob 0.05 \
       --avg-block-length 3 \
       --gpt-min-ratio 0.25 \
       --block-mask-prob 0.1 \
       --short-seq-prob 0.5 \
       --experiment-name blocklm-10b \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 48 \
       --hidden-size 4096 \
       --num-attention-heads 64 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --save /mnt/model_checkpoints \
       --load /cache/blocklm-10b-512 \
       --no-load-lr-scheduler \
       --log-interval 25 \
       --train-iters 250000 \
       --train-data pile cc-news \
       --resume-dataloader \
       --filter-english \
       --loader-scatter 32 \
       --no-lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style linear \
       --lr-decay-ratio 0.1 \
       --lr-decay-iters 50000 \
       --warmup 0.005 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --fp16 \
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"