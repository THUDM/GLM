EXPERIMENT_NAME=${MODEL_TYPE}-wikitext
TASK_NAME=wikitext
DATA_PATH=/root/data/wikitext-103/wiki.test.tokens
EVALUATE_ARGS="--eval-batch-size 16 \
               --seq-length 512 \
               --overlapping-eval 256 \
               --unidirectional"