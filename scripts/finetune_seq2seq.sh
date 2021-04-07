DATA_ROOT=/root/data
CHECKPOINT_PATH=/root/data/checkpoints
SAVE_PATH=/root/data/finetune_checkpoints
DATESTR=$(date +"%m-%d-%H-%M")

source $1    # Model
source $2    # Task

export NCCL_DEBUG=info
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2
export PATH="/opt/conda/bin:$PATH"

NUM_WORKERS=2
NUM_GPUS_PER_WORKER=8
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
HOST_FILE_PATH="/root/code/config/hostfile"

mkdir logs
deepspeed --master_port ${MASTER_PORT} --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} finetune_glm.py \
       --finetune \
       --experiment-name ${EXPERIMENT_NAME} \
       --task ${TASK_NAME} \
       --data-dir ${DATA_PATH} \
       --save ${SAVE_PATH} \
       --checkpoint-activations \
       $MODEL_ARGS \
       $TRAIN_ARGS \
       $COMMON_ARGS \
       $TASK_ARGS \
       2>&1 | tee logs/log-${DATESTR}.txt