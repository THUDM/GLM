#! /bin/bash

# Change for multinode config

NUM_WORKERS=2
NUM_GPUS_PER_WORKER=8
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=bond0 NCCL_IB_GID_INDEX=3 NCCL_NET_GDR_LEVEL=0"
HOST_FILE_PATH="/root/code/config/hostfile"
#OPTIONS_NCCL=""
#HOST_FILE_PATH="/workspace/hostfile"


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


run_cmd="${OPTIONS_NCCL} deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} pretrain_gpt2.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
