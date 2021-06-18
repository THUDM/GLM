DATA_ROOT=/root/data
CHECKPOINT_PATH="/dataset/c07bd62b/finetune_checkpoints"
DATESTR=$(date +"%m-%d-%H-%M")

source $1    # Model
source $2    # Task

export NCCL_DEBUG=info
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export PATH="/opt/conda/bin:$PATH"

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=4
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
HOST_FILE_PATH="/root/code/config/hostfile"

EXPERIMENT_NAME=${EXPERIMENT_NAME}_${DATESTR}
mkdir logs
deepspeed --master_port ${MASTER_PORT} --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} finetune_glm.py \
       --finetune \
       --experiment-name ${EXPERIMENT_NAME} \
       --task ${TASK_NAME} \
       --data-dir ${DATA_PATH} \
       --save ${CHECKPOINT_PATH} \
       --checkpoint-activations \
       --num-workers 1 \
       --no-load-lr-scheduler \
       --fp16 \
       $MODEL_ARGS \
       $TRAIN_ARGS \
       $COMMON_ARGS \
       $TASK_ARGS \
       --model-parallel-size ${MP_SIZE} \
       --overwrite \
       2>&1 | tee logs/log-${EXPERIMENT_NAME}.txt