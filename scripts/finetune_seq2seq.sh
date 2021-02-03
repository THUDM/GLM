source config/model_blocklm_large_generation.sh
EXPERIMENT_NAME=${MODEL_TYPE}-cnndm
CHECKPOINT_PATH="/root/data/checkpoints"

TASK_NAME=cnn_dm
DATA_PATH="/root/data/cnn_dm"

TRAIN_ARGS="--epochs 15 \
            --batch-size 8 \
            --lr 3e-5 \
            --lr-decay-style linear \
            --warmup 0.06 \
            --weight-decay 1.0e-1
            --label-smoothing 0.1"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 1000 \
             --eval-iters 100"

export NCCL_DEBUG=info
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2
export PATH="/opt/conda/bin:$PATH"

NUM_WORKERS=2
NUM_GPUS_PER_WORKER=8
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
HOST_FILE_PATH="/root/code/config/hostfile"

deepspeed --master_port ${MASTER_PORT} --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} finetune_gpt2.py \
       --finetune \
       --experiment-name ${EXPERIMENT_NAME}_608_epoch \
       --task ${TASK_NAME} \
       --data-dir ${DATA_PATH} \
       --save ${CHECKPOINT_PATH} \
       --checkpoint-activations \
       --src-seq-length 608 \
       --tgt-seq-length 160 \
       --min-tgt-length 55 \
       --length-penalty 0.7 \
       --no-repeat-ngram-size 3 \
       --num-beams 5 \
       --select-topk \
       --eval-batch-size 4 \
       $MODEL_ARGS \
       $TRAIN_ARGS \
       $COMMON_ARGS