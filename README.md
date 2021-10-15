# GLM

GLM is a General Language Model pretrained with an autoregressive blank-filling objective and can be finetuned on 
various natural language understanding and generation tasks. 

Please refer to our paper for a detailed description of GLM:

[All NLP Tasks Are Generation Tasks: A General Pretraining Framework](https://arxiv.org/abs/2103.10360)

Zhengxiao Du*, Yujie Qian*, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, Jie Tang (*: equal contribution)

Part of the code is based on [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [PET](https://github.com/timoschick/pet).

## Pretrained Models
You can download the pretrained models used in the paper [here](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/duzx16_mails_tsinghua_edu_cn/Eg8MZe62MlVFs_mK2tHaH-sBC-UC01jpGPZop08pID7sOw?e=MsevNR).

| Name | Params | File | Config
|  -----  | ----  | ---- | ----
| GLM-Base | 110M | glm-base-blank.tar.bz2 | model_blocklm_base.sh
| GLM-Large  | 335M | glm-large-blank.tar.bz2 | model_blocklm_large.sh
| GLM-Large (multi-task) | 335M | glm-large-generation.tar.bz2 | model_blocklm_large_generation.sh
| GLM-410M (multi-task) | 410M | glm-1.25-generation.tar.bz2 | model_blocklm_1.25_generation.sh
| GLM-515M (multi-task) | 515M | glm-1.5-generation.tar.bz2 | model_blocklm_1.5_generation.sh
| GLM-RoBERTa | 335M | glm-roberta-large-blank.tar.bz2 | model_blocklm_roberta_large.sh
| GLM-XXLarge | 10B | [apply here](https://wudaoai.cn/model/download?resourceId=1420992103650996224&filename=GLM-XXLarge-10B-en.tar.bz2) | model_blocklm_10B.sh

## Results

### [SuperGLUE](https://super.gluebenchmark.com)
dev set, single model, single-task finetuning

|  Model | COPA | WSC | RTE | WiC | CB | MultiRC | BoolQ | ReCoRD |
|  ----  | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| GLM-XXLarge  | 98.0 | 95.2 | 93.1 | 75.7 | 98.7/98.2 | 88.1/63.3 | 88.7 | 94.4/94.0 |
| [RoBERTa-Large](https://github.com/pytorch/fairseq/tree/master/examples/roberta) | 94.0 | 91.3 | 86.6 | 75.6 | 98.2/- | 85.7/- | 86.9 |89.5/89.0|
| [DeBERTa-XXLarge-v2](https://github.com/microsoft/DeBERTa/tree/master/experiments/superglue) | 97.0 | - | 93.5 | - | - | 87.8/63.6 | 88.3 | 94.1/93.7 |

### Seq2Seq
[CNN/Daily Mail](https://github.com/abisee/cnn-dailymail) (test set, no additional data used)

|  Model  | ROUGE-1 | ROUGE-2 | ROUGE-L |
|  ----  | ---- | ---- | ---- |
| GLM-XXLarge | **44.7** | 21.4 | **41.4** |
| T5-11B | 43.5 | **21.6** | 40.7 |
| PEGASUS-Large | 44.2 | 21.5 | **41.4** |
| BART-Large | 44.2 | 21.3 | 40.9 |

[XSum](https://github.com/EdinburghNLP/XSum) (test set, no additional data used)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
| ---- | ---- | ---- | ---- |
| GLM-XXLarge | **48.9** | **25.7** | **40.4** |
| PEGASUS-Large | 47.2 | 24.6 | 39.3 |
| BART-Large | 45.1 | 22.3 | 37.3 |

### Language Modeling
test set, zero-shot

| Model | LAMBADA (accuracy) | Wikitext103 (perplexity) |
| ---- | ---- | ---- |
| GLM-XXLarge (bi) | 72.35 | 11.33 |
| GLM-XXLarge (uni) | 67.18 | 12.22 |
| GPT-2 | 52.66 | 17.48 |
| Megatron-LM (8.3B) | 66.51 | 10.81 |
| Turing-NLG | 67.98 | 10.21 |

## Get Started
### Docker Image
We prepare two docker images based on CUDA 10.2 and CUDA 11.2. You can pull the pre-built images from Docker Hub and run with docker v19.03+
  ```shell
  docker run --gpus all --rm -it --ipc=host zxdu20/glm-cuda102
  ```
  or replace `glm-cuda102` with `glm-cuda112`.

  You can also modify the image according to your requirements in [docker/cuda102.dockerfile](docker/cuda102.dockerfile) and build the image yourself
  ```shell
    docker build -f cuda102.dockerfile . -t glm-cuda102
  ```
### Manual Installation 
Please first install PyTorch (we use 1.7.0) and [apex](https://github.com/NVIDIA/apex), and then install other dependencies by `pip install -r requirements.txt`
### Clone this repo
  ```shell
  git clone https://github.com/THUDM/GLM
  cd GLM
  ```

## Usage
We provide scripts for finetuning GLM on some downstream tasks.

### SuperGLUE

- Download the [SuperGlue](https://super.gluebenchmark.com/tasks) data and check the experiment setup in 
  [scripts/ds_finetune_superglue.sh](scripts/ds_finetune_superglue.sh). Note that `DATA_ROOT, CHECKPOINT_PATH, SAVE_PATH` 
  need to be changed to your local path. You may also change the `batch-size` and `nproc_per_node` according to your 
  available hardware.

- Run the following script (use the COPA dataset as an example)

```
bash scripts/ds_finetune_superglue.sh \
     config_tasks/model_blocklm_10B.sh \
     config_tasks/task_copa.sh
```
- We also implement [P-Tuning](https://arxiv.org/abs/2103.10385) in our code. Run the following script to integrate p-tuning:
```shell
bash scripts/ds_finetune_superglue_prompt.sh \
     config_tasks/model_blocklm_10B.sh \
     config_tasks/task_copa.sh
```
  
- To apply GLM to a new NLU dataset with cloze-filling finetuning, implement a `DataProcessor` in
  [tasks/superglue/dataset.py](tasks/superglue/dataset.py) for data loading and add a `PVP` in 
  [tasks/superglue/pvp.py](tasks/superglue/pvp.py) for the cloze question. More details can be found 
  [here](tasks/superglue/README.md).

### Text Summarization

- Download the [Gigaword](https://github.com/harvardnlp/sent-summary), [CNN/Daily Mail](https://github.com/artmatsak/cnn-dailymail) or [XSum](https://github.com/EdinburghNLP/XSum/tree/master/XSum-Dataset) dataset and check the experiment setup in 
  [scripts/ds_finetune_seq2seq.sh](scripts/ds_finetune_seq2seq.sh). Change `DATA_ROOT, CHECKPOINT_PATH, SAVE_PATH` to your 
  local path. 
  
- Run the following script (use the CNN/Daily Mail dataset as an example)

  ```
  bash scripts/ds_finetune_seq2seq.sh \ 
     config_tasks/model_blocklm_10B.sh \ 
     config_tasks/seq_cnndm_org.sh
  ```
- The summaries are written into `./runs/experiment_name/test.jsonl.hyps`. The references are written into `test.jsonl.refs` in the same directory. For calculating rouge, install [file2rouge](https://github.com/pltrdy/files2rouge) and download Stanford CoreNLP from [here](http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip). Run  the following script
  ```
  bash scripts/evaluate_seq2seq.sh \
   ./runs/experiment_name/test.jsonl.hyps ./runs/experiment_name/test.jsonl.refs
  ```

### Language Modeling
#### LAMBADA Cloze Accuracy
* Download the [LAMBADA](https://github.com/cybertronai/bflm/blob/master/lambada_test.jsonl) data and change 
  `DATA_ROOT, CHECKPOINT_PATH` in [scripts/evaluate_lm.sh](scripts/evaluate_lm.sh)
* Run the following script
```shell
bash scripts/evaluate_lm.sh \ 
     config_tasks/model_blocklm_large_generation.sh \
     config_tasks/zero_lambada.sh 
```
#### LM Perplexity
* Download our [test set of wikibook](https://mailstsinghuaeducn-my.sharepoint.com/:t:/g/personal/duzx16_mails_tsinghua_edu_cn/EQa_B6KY_q1FjtUeG-T52iMBFtNrfhfHcZbzMxfkJKXKRQ?e=inTdHh) or [Wikitext103](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip) dataset and change `DATA_ROOT, CHECKPOINT_PATH` 
  in [scripts/evaluate_lm.sh](scripts/evaluate_lm.sh)
* Run the following script
  ```shell
  bash scripts/evaluate_lm.sh \ 
     config_tasks/model_blocklm_large_generation.sh \
     config_tasks/zero_wikitext.sh 
  ```

### Blank Language Model
- Download the [Yahoo](https://github.com/Varal7/blank_language_model) dataset and check the experiment setup in 
  [scripts/finetune_blank.sh](scripts/finetune_blank.sh). Change `DATA_ROOT, CHECKPOINT_PATH, SAVE_PATH` to your 
  local path. 
  
- Run the following script

```
bash scripts/finetune_blank.sh \ 
     config_tasks/model_blocklm_large.sh \ 
     config_tasks/seq_blank.sh
```

### Blank Filling (Interactive)
* Change `CHECKPOINT_PATH` to your local path. Run the following script
```
bash scripts/generate_block.sh \
     config_tasks/model_blocklm_large.sh
```
Example:

Context: Ng is an adjunct professor at [MASK] (formerly associate professor and Director of its Stanford AI Lab or SAIL ). Also a pioneer in online education, Ng co-founded Coursera and deeplearning.ai.

GLM: [CLS] ng is an adjunct professor at [MASK] ( formerly associate professor and director of its stanford ai lab or sail ) . also a pioneer in online education , ng co - founded coursera and deeplearning . ai . [PAD] <|startofpiece|> the stanford university

### Model Parallelism
If your encounter the `CUDA out of memory` error, which means you GPU memory is limited, you can try the model parallelism to divide the parameters into multiple GPUs. Take the two-way model parallelism as an example. First run `change_mp.py` to divide the checkpoint:
```shell
python change_mp.py path_to_the_checkpoint 2
```
Then update the checkpoint path in the model config file (such as [config_tasks/model_blocklm_10B.sh](config_tasks/model_blocklm_10B.sh)) and change `MP_SIZE` in the script (such as [scripts/ds_finetune_superglue.sh](scripts/ds_finetune_superglue.sh)) to `2`.

## Pretrain
Run the following script to pre-train the GLM-Large model
```shell
bash scripts/ds_pretrain_nvidia.sh config/ds_block_large.sh
```

The script [scripts/ds_pretrain_nvidia.sh](scripts/ds_pretrain_nvidia.sh) launches the training program with DeepSpeed. You should change `NUM_WORKERS` and `NUM_GPUS_PER_WORKER` to the number of workers and the number of gpus per worker. Also change `HOST_FILE_PATH` to the path to an OpenMPI-style hostfile. More details about DeepSpeed launcher can be found [here](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node).

The file [config/ds_block_large.sh](config/ds_block_large.sh) defines the hyperparameters for pretraining. Most of the arguments are fairly self-explanatory. Specifically, `--train-data` can be multiple keywords defined in `NAMED_CORPORA` in [data_utils/corpora.py](data_utils/corpora.py). The hyperparameters of the optimizer are defined in the corresponding json file under `config`. The semantics of the json file can be found [here](https://www.deepspeed.ai/docs/config-json).

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
```