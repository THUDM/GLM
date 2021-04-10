DATA_ROOT=/root/data/superglue
source config_tasks/model_blocklm_base_na.sh
source $1

CHECKPOINT_PATH="/root/data/finetune_checkpoints"

N_GPU=2
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
DISTRIBUTED_ARGS="--nproc_per_node ${N_GPU} --nnodes 1 --node_rank 0 --master_addr localhost --master_port $MASTER_PORT"

DATESTR=$(date +"%m-%d-%H-%M")
GRID_LOG=logs/grid_${EXPERIMENT_NAME}_${DATESTR}.txt

for lr in 6e-5 1e-5 2e-5
do
    for seed in 1234 5678 9753
    do
    HYPER=${lr}-${seed}
    PER_GPU_BS=$((bs/N_GPU))
    python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune_glm.py \
       --finetune \
       --experiment-name ${EXPERIMENT_NAME}/${HYPER} \
       --task ${TASK_NAME} \
       --data-dir ${DATA_PATH} \
       --save ${CHECKPOINT_PATH} \
       --checkpoint-activations \
       --seq-length ${MAX_SEQ_LEN} \
       --eval-batch-size 16 \
       $MODEL_ARGS \
       $COMMON_ARGS \
       --lr ${lr} \
       --batch-size 8 \
       --seed ${seed} \
       2>&1 | tee logs/log-${EXPERIMENT_NAME}-${HYPER}.txt
    echo $lr $bs $epoch $warmup $seed >> $GRID_LOG
    cat runs/${EXPERIMENT_NAME}/${HYPER}/results.json >> $GRID_LOG
    done
done

echo $EXPERIMENT_NAME >> $GRID_LOG