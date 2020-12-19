GPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

TRAIN_DATA="/root/data/RACE/train/middle /root/data/RACE/train/high"
VALID_DATA="/root/data/RACE/dev/middle /root/data/RACE/dev/high"
TEST_DATA="/root/data/RACE/test/middle /root/data/RACE/test/high"
PRETRAINED_CHECKPOINT=/root/data/checkpoints/block-lm-blank12-11-05-38
CHECKPOINT_PATH=/root/data/checkpoints
EXPERIMENT_NAME=block-lm-blank
COMMON_TASK_ARGS="--block-lm \
                  --num-layers 12 \
                  --hidden-size 768 \
                  --num-attention-heads 12 \
                  --seq-length 512 \
                  --max-position-embeddings 512 \
                  --fp16 \
                  --tokenizer-model-type bert-base-uncased"

COMMON_TASK_ARGS_EXT="--train-data $TRAIN_DATA \
                      --valid-data $VALID_DATA \
                      --test-data $TEST_DATA \
                      --load-pretrained $PRETRAINED_CHECKPOINT \
                      --checkpoint-activations \
                      --save-interval 10000 \
                      --save $CHECKPOINT_PATH \
                      --log-interval 100 \
                      --eval-interval 1000 \
                      --eval-iters 100 \
                      --weight-decay 1.0e-1"

#MASTER_PORT=${MASTER_PORT} python finetune_gpt2.py \
python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune_gpt2.py \
       --experiment-name ${EXPERIMENT_NAME} \
       --task RACE \
       --finetune \
       $COMMON_TASK_ARGS \
       $COMMON_TASK_ARGS_EXT \
       --tokenizer-type BertWordPieceTokenizer \
       --pool-token pad \
       --epochs 5 \
       --batch-size 4 \
       --lr 1.5e-5 \
       --lr-decay-style linear \
       --warmup 0.06