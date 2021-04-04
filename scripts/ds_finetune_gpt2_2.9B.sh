#! /bin/bash

# Change for multinode config

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=bond0 NCCL_IB_GID_INDEX=3 NCCL_NET_GDR_LEVEL=0"
HOST_FILE_PATH="/root/code/config/pre_hostfile"
#OPTIONS_NCCL=""
#HOST_FILE_PATH="/workspace/hostfile"


config_json="$script_dir/ds_config_29B.json"
gpt_options=" \
       --finetune \
       --experiment-name txl-2.8b \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 32 \
       --hidden-size 2560 \
       --num-attention-heads 32 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --mem-length 256 \
       --load /root/data/checkpoints/txl-2.8b11-20-15-10 \
       --no-load-optim \
       --save /root/data/finetune_checkpoints \
       --save-interval 2000 \
       --train-iters 10000 \
       --resume-dataloader \
       --train-data /root/data/finetune_data.json \
       --lazy-loader \
       --tokenizer-type ChineseSPTokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --lr-decay-ratio 0.01 \
       --lr-decay-iters 10000 \
       --warmup .1 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --transformer-xl \
       --fp16 \
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"


run_cmd="${OPTIONS_NCCL} deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} pretrain_gpt2.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
