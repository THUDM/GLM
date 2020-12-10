#!/bin/bash

CHECKPOINT_PATH=/root/data/checkpoints/block-lm-base12-09-15-45
MPSIZE=1
NLAYERS=12
NHIDDEN=768
NATT=12
MAXSEQLEN=512
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

#SAMPLING ARGS
TEMP=1.0
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=0
TOPP=0

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

config_json="$script_dir/ds_config.json"

MASTER_PORT=${MASTER_PORT} python generate_samples.py \
       --block-lm \
       --model-parallel-size $MPSIZE \
       --deepspeed_config ${config_json} \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
       --load $CHECKPOINT_PATH \
       --num-attention-heads $NATT \
       --max-position-embeddings 512 \
       --tokenizer-type BertWordPieceTokenizer \
       --tokenizer-model-type bert-base-uncased \
       --fp16 \
       --cache-dir cache \
       --out-seq-length $MAXSEQLEN \
       --seq-length 512 \
       --temperature $TEMP \
       --top_k $TOPK \
       --top_p $TOPP
