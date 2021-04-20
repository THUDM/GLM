DATA_ROOT=/dataset/fd5061f6/english_data
CHECKPOINT_PATH="/root/data/checkpoints"

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port $MASTER_PORT"
DATESTR=$(date +"%m-%d-%H-%M")

source config_tasks/model_blocklm_10B.sh
source $1

python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune_gpt2.py \
       --deepspeed \
       --finetune \
       --experiment-name ${EXPERIMENT_NAME} \
       --task ${TASK_NAME} \
       --valid-data ${DATA_PATH} \
       --save ${CHECKPOINT_PATH} \
       --checkpoint-activations \
       --fp16 \
       $MODEL_ARGS \
       $EVALUATE_ARGS \
       2>&1 | tee logs/log-${DATESTR}.txt