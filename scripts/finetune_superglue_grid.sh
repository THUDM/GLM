DATA_ROOT=/root/data/superglue
source config_tasks/model_blocklm_roberta_large.sh
source $1

CHECKPOINT_PATH="/root/data/finetune_checkpoints"

N_GPU=2
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
DISTRIBUTED_ARGS="--nproc_per_node ${N_GPU} --nnodes 1 --node_rank 0 --master_addr localhost --master_port $MASTER_PORT"

DATESTR=$(date +"%m-%d-%H-%M")
GRID_LOG=logs/grid_${EXPERIMENT_NAME}_${DATESTR}.txt

for lr in 1e-5 #2e-5
do
  for bs in 16 #32
  do
    for epoch in 40 #10 20 40
    do
    for warmup in 0.1 #0.06 0
    do
    for wd in 0.1 0.01 0
    do
    for beta2 in 0.98 # 0.999
    do
    for eps in 1e-6 # 1e-8
    do
    for seed in 1 2 3 # 4 5
    do
    HYPER=${lr}-b${bs}-ep${epoch}-wm${warmup}-wd${wd}-${seed}
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
       $COMMON_ARGS \
       --lr-decay-style linear \
       --epochs ${epoch} \
       --lr ${lr} \
       --weight-decay ${wd} \
       --warmup ${warmup} \
       --batch-size ${PER_GPU_BS} \
       --seed ${seed} \
       --optimizer adam \
       --adam-beta2 ${beta2} \
       --adam-eps ${eps} \
       --overwrite \
       2>&1 | tee logs/log-${EXPERIMENT_NAME}-${HYPER}.txt
    echo $lr $bs $epoch $warmup $seed >> $GRID_LOG
    cat runs/${EXPERIMENT_NAME}/${HYPER}/results.json >> $GRID_LOG
    done
    done
    done
    done
    done
    done
  done
done

echo $EXPERIMENT_NAME >> $GRID_LOG
