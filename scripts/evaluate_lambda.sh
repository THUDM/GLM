CHECKPOINT_PATH="/root/data/checkpoints"

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port $MASTER_PORT"
DATESTR=$(date +"%m-%d-%H-%M")

MAX_SEQ_LEN=512
source config/model_gpt_large.sh

#EXPERIMENT_NAME=${MODEL_TYPE}-lambda_uni
#TASK_NAME=lambda
#DATA_PATH=/root/data/lambada_test.jsonl

#EXPERIMENT_NAME=${MODEL_TYPE}-wikitext
#TASK_NAME=wikitext
#DATA_PATH=/root/data/wikitext-103/wiki.test.tokens

EXPERIMENT_NAME=${MODEL_TYPE}-lm_uni
TASK_NAME=language_model
DATA_PATH=/root/data/bert-large-test.txt

python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune_gpt2.py \
       --finetune \
       --experiment-name ${EXPERIMENT_NAME} \
       --task ${TASK_NAME} \
       --valid-data ${DATA_PATH} \
       --save ${CHECKPOINT_PATH} \
       --checkpoint-activations \
       --seq-length ${MAX_SEQ_LEN} \
       --eval-batch-size 16 \
       --overlapping-eval 256 \
       --unidirectional \
       $MODEL_ARGS \
       2>&1 | tee logs/log-${DATESTR}.txt