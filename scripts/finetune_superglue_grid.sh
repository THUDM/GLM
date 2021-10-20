DATA_ROOT="/dataset/fd5061f6/english_data/superglue"
CHECKPOINT_PATH="/dataset/fd5061f6/english_data/checkpoints"
SAVE_PATH="/dataset/fd5061f6/finetune_checkpoints"
DATESTR=$(date +"%m-%d-%H-%M")

source $1    # Model
source $2    # Task

if [ -z $N_GPU ];then
  N_GPU=2
fi
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
DISTRIBUTED_ARGS="--nproc_per_node ${N_GPU} --nnodes 1 --node_rank 0 --master_addr localhost --master_port $MASTER_PORT"

GRID_LOG=logs/grid_${EXPERIMENT_NAME}_${DATESTR}.txt
mkdir logs
for lr in 5e-6 1e-5 2e-5
do
  for seed in 1234 5678 3456
  do
  HYPER=${lr}-${seed}
  PER_GPU_BS=$(($BATCH_SIZE/$N_GPU))
  if [ ! -f runs/${EXPERIMENT_NAME}/${HYPER}/test_results.json ]; then
    echo runs/${EXPERIMENT_NAME}/${HYPER}
    python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune_glm.py \
       --finetune \
       --experiment-name ${EXPERIMENT_NAME}/${HYPER} \
       --task ${TASK_NAME} \
       --data-dir ${DATA_PATH} \
       --save ${CHECKPOINT_PATH} \
       --seq-length ${MAX_SEQ_LEN} \
       --checkpoint-activations \
       --eval-batch-size 16 \
       --save-epoch 100000 \
       --no-load-optim \
       --no-load-lr-scheduler \
       $MODEL_ARGS \
       $TRAIN_ARGS \
       $COMMON_ARGS \
       --fp16 \
       --batch-size ${PER_GPU_BS} \
       --epochs ${EPOCH_SINGLE} \
       --lr ${lr} \
       --seed ${seed} \
       --overwrite \
       2>&1 | tee logs/log-${EXPERIMENT_NAME}-${HYPER}.txt
  fi
  echo $lr $seed >> $GRID_LOG
  cat runs/${EXPERIMENT_NAME}/${HYPER}/results.json >> $GRID_LOG
  done
done

echo $EXPERIMENT_NAME >> $GRID_LOG