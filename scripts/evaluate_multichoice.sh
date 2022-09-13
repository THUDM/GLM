CHECKPOINT_PATH=<Your checkpoint directory>
DATA_PATH= <Your jsonl file path>

source $1    # Model

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=1
MP_SIZE=1
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
MAX_SEQ_LEN=512

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
DISTRIBUTED_ARGS="${OPTIONS_NCCL} deepspeed --master_port $MASTER_PORT --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER}"

mkdir logs
run_cmd="${DISTRIBUTED_ARGS} finetune_glm.py \
       --deepspeed \
       --deepspeed_config config_tasks/config_blocklm_10B.json \
       --finetune \
       --cloze-eval \
       --task multichoice \
       --test-data ${DATA_PATH} \
       --seq-length ${MAX_SEQ_LEN} \
       --checkpoint-activations \
       --eval-batch-size 16 \
       --num-workers 1 \
       --no-load-optim \
       --no-load-lr-scheduler \
       $MODEL_ARGS \
       --fp16 \
       --model-parallel-size ${MP_SIZE} \
       --epochs 0 \
       --overwrite \
       2>&1"

echo ${run_cmd}
eval ${run_cmd}
