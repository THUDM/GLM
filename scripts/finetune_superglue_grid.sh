source config_tasks/model_blocklm.sh
source $1

CHECKPOINT_PATH="/root/data/checkpoints"

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
DISTRIBUTED_ARGS="--nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr localhost --master_port $MASTER_PORT"

mkdir logs
for lr in ${LR_RANGE[@]}
do
  for epoch in ${EPOCH_RANGE[@]}
  do
    DATESTR=$(date +"%m-%d-%H-%M")
    python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune_gpt2.py \
       --finetune \
       --experiment-name ${EXPERIMENT_NAME}/${lr}_${epoch} \
       --task ${TASK_NAME} \
       --data-dir ${DATA_PATH} \
       --save ${CHECKPOINT_PATH} \
       --checkpoint-activations \
       --seq-length ${MAX_SEQ_LEN} \
       --eval-batch-size 16 \
       $MODEL_ARGS \
       $TRAIN_ARGS \
       $COMMON_ARGS \
       --epochs ${epoch} \
       --lr ${lr} \
       2>&1 | tee logs/log-${EXPERIMENT_NAME}-${lr}_${epoch}-${DATESTR}.txt
  done
done