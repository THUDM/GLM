source $1
source scripts/superglue_args.sh

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
DISTRIBUTED_ARGS="--nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port $MASTER_PORT"
DATESTR=$(date +"%m-%d-%H-%M")

mkdir logs
python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune_gpt2.py ${finetune_options} 2>&1 | tee logs/log-${DATESTR}.txt