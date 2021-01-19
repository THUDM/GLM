CHECKPOINT_PATH="/root/data/checkpoints"

finetune_options="--finetune \
       --experiment-name ${EXPERIMENT_NAME} \
       --task ${TASK_NAME} \
       --data-dir ${DATA_PATH} \
       --save ${CHECKPOINT_PATH} \
       --checkpoint-activations \
       --seq-length ${MAX_SEQ_LEN} \
       --eval-batch-size 16 \
       $MODEL_ARGS \
       $TRAIN_ARGS \
       $COMMON_ARGS \
"