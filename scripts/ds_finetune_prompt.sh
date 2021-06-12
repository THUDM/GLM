#! /bin/bash
DATESTR=$(date +"%m-%d-%H-%M")

#mkdir /cache
#cp -r /mnt/model_checkpoints/blocklm-10b-latest /cache

if [[ $DLS_TASK_NUMBER == 1 ]]; then
    MASTER_ADDR=localhost
    MASTER_PORT=$(shuf -n 1 -i 10000-65535)
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

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0"

cd $main_dir

MP_SIZE=1
DATA_ROOT=/mnt/superglue
source config_tasks/model_blocklm_10B.sh
source $1

CHECKPOINT_PATH="/mnt/finetune_checkpoints"

EXPERIMENT_NAME=${EXPERIMENT_NAME}_ptuning-$2_${DATESTR}

mkdir logs
if [[ $DLS_TASK_NUMBER == 1 ]]; then
  LOG_FILE=logs/log-${EXPERIMENT_NAME}.txt
else
  LOG_FILE=logs/log-${EXPERIMENT_NAME}-${NODE_RANK}.txt
fi

run_cmd="${OPTIONS_NCCL} python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune_gpt2.py \
       --deepspeed \
       --deepspeed_config config_tasks/config_blocklm_10B.json \
       --finetune \
       --experiment-name ${EXPERIMENT_NAME} \
       --task ${TASK_NAME} \
       --data-dir ${DATA_PATH} \
       --save ${CHECKPOINT_PATH} \
       --seq-length ${MAX_SEQ_LEN} \
       --checkpoint-activations \
       --eval-batch-size 16 \
       --save-epoch 10000 \
       --num-workers 1 \
       --no-load-optim \
       --no-load-lr-scheduler \
       --fp16 \
       $MODEL_ARGS \
       $TRAIN_ARGS \
       $COMMON_ARGS \
       --continuous-prompt \
       --pattern-id $2 \
       --num-prompt-tokens 3 \
       --model-parallel-size ${MP_SIZE} \
       --epochs ${EPOCH_SINGLE} \
       --overwrite \
       2>&1 | tee ${LOG_FILE}"

echo ${run_cmd}
eval ${run_cmd}
