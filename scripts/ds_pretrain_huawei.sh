#! /bin/bash
DATESTR=$(date +"%m-%d-%H-%M")

mkdir /cache
cp -r /mnt/model_checkpoints/blocklm-10b-latest /cache

if [[ $DLS_TASK_NUMBER == 1 ]]; then
    MASTER_ADDR=localhost
    MASTER_PORT=6000
    NNODES=1
    NODE_RANK=0
else
    MASTER_HOST="$BATCH_CUSTOM0_HOSTS"
    MASTER_ADDR="${MASTER_HOST%%:*}"
    MASTER_PORT="${MASTER_HOST##*:}"
    NNODES="$DLS_TASK_NUMBER"
    NODE_RANK="$DLS_TASK_INDEX"
fi

GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
MP_SIZE=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

cd $main_dir
source $1

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0"

mkdir logs
mkdir logs/${DATESTR}
run_cmd="${OPTIONS_NCCL} python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt2.py ${gpt_options} 2>&1 | tee logs/${DATESTR}/log-${NODE_RANK}.txt"
echo ${run_cmd}
eval ${run_cmd}

set +x