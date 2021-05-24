# GLM

GLM is a General Language Model pretrained with an autoregressive blank-filling objective and can be finetuned on 
various natural language understanding and generation tasks. 

Please refer to our paper for a detailed description of GLM:

[All NLP Tasks Are Generation Tasks: A General Pretraining Framework](https://arxiv.org/abs/2103.10360)

Zhengxiao Du*, Yujie Qian*, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, Jie Tang (*: equal contribution)

Part of the code is based on [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [PET](https://github.com/timoschick/pet).


## Pretrain
First change `PATH` of the corresponding class (`Pile` and `CCNews`) in `NAMED_CORPORA` at [data_utils/corpora.py](data_utils/corpora.py). Then run
```shell
bash scripts/ds_pretrain_nvidia.sh config/ds_block_2B.sh
```
The script [scripts/ds_pretrain_nvidia.sh](scripts/ds_pretrain_nvidia.sh) launch the training program with DeepSpeed. You should change `NUM_WORKERS` and `NUM_GPUS_PER_WORKER` to the number of workers and the number of gpus per worker. Also change `` to the path to an [OpenMPI-style hostfile]((https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node)).

## Citation
Please cite our paper if you find this code useful for your research:
```
@article{DBLP:journals/corr/abs-2103-10360,
  author    = {Zhengxiao Du and
               Yujie Qian and
               Xiao Liu and
               Ming Ding and
               Jiezhong Qiu and
               Zhilin Yang and
               Jie Tang},
  title     = {All {NLP} Tasks Are Generation Tasks: {A} General Pretraining Framework},
  journal   = {CoRR},
  volume    = {abs/2103.10360},
  year      = {2021},
  url       = {https://arxiv.org/abs/2103.10360}
}
``