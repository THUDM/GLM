EXPERIMENT_NAME=${MODEL_TYPE}-lambda
TASK_NAME=lambda
DATA_PATH=/root/data/lambada_test.jsonl
EVALUATE_ARGS="--eval-batch-size 16 \
               --seq-length 512"