CUDA_VISIBLE_DEVICES=5 python inference.py --input_dir "./data/kqapro/processed" --output_dir "./result/kqapro_inference/" --config "./data/kqapro/config.py" --model_name_or_path "/data0/sjd/HW3/glm-roberta-large" --ckpt "./result/kqapro/checkpoint-best"