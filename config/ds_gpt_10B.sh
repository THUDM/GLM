#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

config_json="$script_dir/config_gpt_10B.json"
gpt_options=" \
       --experiment-name txl-10b \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 48 \
       --hidden-size 4096 \
       --num-attention-heads 32 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --save /dataset/fd5061f6/english_data/checkpoints \
       --log-interval 25 \
       --eval-interval 250 \
       --save-interval 1000 \
       --train-iters 100000 \
       --resume-dataloader \
       --train-data wikibook \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --lr-decay-ratio 0.1 \
       --lr-decay-iters 20000 \
       --warmup .2 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --fp16 \
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"