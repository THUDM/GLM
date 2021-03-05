DATA_ROOT=/root/data/superglue
source config_tasks/model_blocklm_roberta_large.sh
source $1

CHECKPOINT_PATH="/root/data/finetune_checkpoints"

N_GPU=2
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
DISTRIBUTED_ARGS="--nproc_per_node ${N_GPU} --nnodes 1 --node_rank 0 --master_addr localhost --master_port $MASTER_PORT"

GRID_LOG=logs/grid_${EXPERIMENT_NAME}.txt
echo $EXPERIMENT_NAME > $GRID_LOG

for lr in 5e-6 1e-5 2e-5
do
  for bs in 16 32
  do
    for seed in 1 2 3 4 5
    do
    HYPER=${lr}-${bs}-${seed}
    DATESTR=$(date +"%m-%d-%H-%M")
    PER_GPU_BS=$((bs/N_GPU))
    python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune_gpt2.py \
       --finetune \
       --experiment-name ${EXPERIMENT_NAME}/${HYPER} \
       --task ${TASK_NAME} \
       --data-dir ${DATA_PATH} \
       --save ${CHECKPOINT_PATH} \
       --checkpoint-activations \
       --seq-length ${MAX_SEQ_LEN} \
       --eval-batch-size 16 \
       $MODEL_ARGS \
       $TRAIN_ARGS \
       $COMMON_ARGS \
       --epochs ${EPOCH_SINGLE} \
       --lr ${lr} \
       --batch-size ${PER_GPU_BS} \
       --seed ${seed} \
       --optimizer adam \
       --overwrite \
       2>&1 | tee logs/log-${EXPERIMENT_NAME}-${HYPER}.txt
    echo $lr $bs $seed >> $GRID_LOG
    cat runs/${EXPERIMENT_NAME}/${HYPER}/results.json >> $GRID_LOG
    done
  done
done