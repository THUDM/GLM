CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --rdzv_backend=c10d --max_restarts 0 --rdzv_endpoint=localhost:12369 --nproc_per_node=4 -m train --input_dir "/data/sjd/HW3/data/kqapro/processed" --output_dir "/data/sjd/HW3/result/kqapro" --config "/data/sjd/HW3/data/kqapro/config.py" --model_name_or_path "/data/sjd/HW3/glm-roberta-large"