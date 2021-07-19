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
GRID_LOG=logs/grid_${EXPERIMENT_NAME}_${DATESTR}.txt

for lr in 5e-6 1e-5 2e-5
do
  for batch in 4 8 12
  do
    for epoch in 5 10
    do
      HYPER=${lr}-${batch}-${epoch}
      python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune_glm.py \
             --finetune \
             --experiment-name ${EXPERIMENT_NAME}/${HYPER} \
             --task ${TASK_NAME} \
             --data-dir ${DATA_PATH} \
             --save ${SAVE_PATH} \
             --checkpoint-activations \
             --epochs ${epoch} \
             --batch-size ${batch} \
             --lr ${lr} \
             $MODEL_ARGS \
             $TRAIN_ARGS \
             $COMMON_ARGS \
             $TASK_ARGS \
             2>&1 | tee logs/log-${EXPERIMENT_NAME}-${HYPER}.txt
      echo $lr $batch $epoch >> $GRID_LOG
      cat runs/${EXPERIMENT_NAME}/${HYPER}/results.json >> $GRID_LOG
    done
  done
done

