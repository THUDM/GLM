#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

config_json="$script_dir/config_gpt_large.json"
gpt_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 24 \
       --hidden-size 1536 \
       --num-attention-heads 16 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --save /root/data/checkpoints \
       --train-iters 50000 \
       --resume-dataloader \
       --train-data wikibook \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .01 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --fp16 \
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"