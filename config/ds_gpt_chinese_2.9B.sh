#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

config_json="$script_dir/ds_config_29B.json"
gpt_options=" \
       --experiment-name txl-2.8b \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 32 \
       --hidden-size 2560 \
       --num-attention-heads 32 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --mem-length 256 \
       --load /root/data/checkpoints/txl-2.8b11-20-15-10 \
       --save /root/data/checkpoints \
       --save-interval 2000 \
       --train-iters 300000 \
       --resume-dataloader \
       --train-data zhihu baike zhidao \
       --lazy-loader \
       --tokenizer-type ChineseSPTokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --lr-decay-ratio 0.1 \
       --lr-decay-iters 300000 \
       --warmup .01 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --transformer-xl \
       --fp16 \
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"
