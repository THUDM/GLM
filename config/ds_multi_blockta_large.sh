#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

config_json="$script_dir/config_blockta_multi_large.json"
gpt_options=" \
       --block-lm \
       --task-mask \
       --bert-prob 0.5 \
       --gap-sentence-prob 0.3 \
       --avg-block-length 3 \
       --gpt-min-ratio 0.25 \
       --block-mask-prob 0.1 \
       --short-seq-prob 0.02 \
       --experiment-name blocklm-roberta-large-multi \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --seq-length 512 \
       --max-sequence-length 1025 \
       --save /dataset/fd5061f6/english_data/checkpoints \
       --log-interval 50 \
       --eval-interval 1000 \
       --save-interval 5000 \
       --train-iters 500000 \
       --train-data multilingual \
       --dataset-temperature 0.3 \
       --loader-scatter 32 \
       --loader-fraction 0.1 \
       --resume-dataloader \
       --no-pre-tokenize \
       --tokenizer-type ChineseSPTokenizer \
       --tokenizer-model-type /dataset/fd5061f6/duzx16/tokenizer/mglm-unigram-250k/mglm250k-uni.model \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style linear \
       --lr-decay-ratio 0.1 \
       --lr-decay-iters 400000 \
       --warmup 0.02 \
       --checkpoint-activations \
       --fp16 \
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"