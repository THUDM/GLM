#!/bin/bash

CHECKPOINT_PATH=/data/checkpoints/txl-2.8b11-20-15-10
MPSIZE=1
NLAYERS=32
NHIDDEN=2560
NATT=32
MAXSEQLEN=1024
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

#SAMPLING ARGS
TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=0
TOPP=0

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

config_json="$script_dir/ds_config.json"

deepspeed --num_nodes 1 --num_gpus 1 --master_port ${MASTER_PORT} generate_samples.py \
       --deepspeed \
       --model-parallel-size $MPSIZE \
       --deepspeed_config ${config_json} \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
       --load $CHECKPOINT_PATH \
       --num-attention-heads $NATT \
       --max-position-embeddings 1024 \
       --tokenizer-type ChineseSPTokenizer \
       --fp16 \
       --cache-dir cache \
       --out-seq-length $MAXSEQLEN \
       --seq-length 512 \
       --mem-length 256 \
       --transformer-xl \
       --temperature $TEMP \
       --top_k $TOPK \
       --top_p $TOPP
