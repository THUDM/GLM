#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

config_json="$script_dir/ds_config.json"
gpt_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 40 \
       --hidden-size 1408 \
       --num-attention-heads 16 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --load /root/data/checkpoints/gpt-345M11-14-13-21 \
       --save /root/data/checkpoints \
       --save-interval 2000 \
       --train-iters 320000 \
       --resume-dataloader \
       --train-data zhihu baike zhidao \
       --lazy-loader \
       --tokenizer-type ChineseSPTokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-ratio 0.1 \
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
