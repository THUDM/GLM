GPUS_PER_NODE=1
MASTER_ADDR=localhost
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
DATESTR=$(date +"%m-%d-%H-%M")
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

DATA_PATH="/root/data/superglue/ReCoRD"
CHECKPOINT_PATH="/root/data/checkpoints"
EXPERIMENT_NAME=blank-base
PRETRAINED_CHECKPOINT=/root/data/checkpoints/block-lm-blank-cls12-18-12-50
CHECKPOINT=/root/data/checkpoints/blank-base-record-ReCoRD-12-29-12-22

MODEL_ARGS="--block-lm \
            --cloze-eval \
            --num-layers 12 \
            --hidden-size 768 \
            --num-attention-heads 12 \
            --seq-length 512 \
            --max-position-embeddings 512 \
            --tokenizer-model-type bert-base-uncased \
            --tokenizer-type BertWordPieceTokenizer \
            --load-pretrained $PRETRAINED_CHECKPOINT"

TRAIN_ARGS="--epochs 5 \
            --batch-size 8 \
            --lr 1e-5 \
            --lr-decay-style linear \
            --warmup 0.06 \
            --weight-decay 1.0e-1"

COMMON_ARGS="--checkpoint-activations \
             --save-interval 10000 \
             --save $CHECKPOINT_PATH \
             --log-interval 100 \
             --eval-interval 1000 \
             --eval-iters 100"

mkdir logs
python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune_gpt2.py \
       --experiment-name ${EXPERIMENT_NAME} \
       --task ReCoRD \
       --finetune \
       --data-dir $DATA_PATH \
       $MODEL_ARGS \
       $TRAIN_ARGS \
       $COMMON_ARGS \
       2>&1 | tee logs/log-${DATESTR}.txt