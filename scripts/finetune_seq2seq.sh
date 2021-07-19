DATA_ROOT=/root/data
CHECKPOINT_PATH=/dataset/fd5061f6/pretrained_models
SAVE_PATH=/dataset/fd5061f6/finetune_checkpoints
DATESTR=$(date +"%m-%d-%H-%M")

source $1    # Model
source $2    # Task

if [ -z $N_GPU ];then
  N_GPU=4
fi

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
DISTRIBUTED_ARGS="--nproc_per_node ${N_GPU} --nnodes 1 --node_rank 0 --master_addr localhost --master_port $MASTER_PORT"

DATESTR=$(date +"%m-%d-%H-%M")
EXPERIMENT_NAME=${EXPERIMENT_NAME}  #-${DATESTR}

TOKENIZERS_PARALLELISM=false

mkdir logs
python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune_glm.py \
       --finetune \
       --experiment-name ${EXPERIMENT_NAME} \
       --task ${TASK_NAME} \
       --data-dir ${DATA_PATH} \
       --save ${SAVE_PATH} \
       --checkpoint-activations \
       --epochs ${EPOCH_SINGLE} \
       --batch-size ${BATCH_SINGLE} \
       --lr ${LR_SINGLE} \
       $MODEL_ARGS \
       $TRAIN_ARGS \
       $COMMON_ARGS \
       $TASK_ARGS \
       2>&1 | tee logs/log-${EXPERIMENT_NAME}.txt

