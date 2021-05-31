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
| GLM-XXLarge | 10B | to be released | to be released
## Installation
Clone this repo
```shell
git clone https://github.com/THUDM/GLM
cd GLM
```
Please first install PyTorch (we use 1.7.0) and [apex](https://github.com/NVIDIA/apex), and then install other dependencies by
```shell
pip install -r requirements.txt
```

## Usage
We provide scripts for finetuning GLM on some downstream tasks.

### SuperGLUE

- Download the [SuperGlue](https://super.gluebenchmark.com/tasks) data and check the experiment setup in 
  [scripts/finetune_superglue.sh](scripts/finetune_superglue.sh). Note that `DATA_ROOT, CHECKPOINT_PATH, SAVE_PATH` 
  need to be changed to your local path. You may also change the `batch-size` and `nproc_per_node` according to your 
  available hardware. We suggest to use aggregated batch size 64 for `MultiRC` and `ReCORD` and 16 for other tasks.

- Run the following script (use the COPA dataset as an example)

```
bash scripts/finetune_superglue.sh \
     config_tasks/model_blocklm_roberta_large.sh \
     config_tasks/task_copa.sh
```

- To apply GLM to a new NLU dataset with cloze-filling finetuning, implement a `DataProcessor` in
  [tasks/superglue/dataset.py](tasks/superglue/dataset.py) for data loading and add a `PVP` in 
  [tasks/superglue/pvp.py](tasks/superglue/pvp.py) for the cloze question. More details can be found 
  [here](tasks/superglue/README.md).
  
- The cloze questions (prompts) used in this work are written by human. We are also studying a P-tuning (prompt 
  tuning) approach to search for the optimal continuous prompt. Please refer to our 
  [paper](https://arxiv.org/abs/2103.10385) and [code](https://github.com/THUDM/P-tuning).

### Text Summarization

- Download the [Gigaword](https://github.com/harvardnlp/sent-summary) dataset and check the experiment setup in 
  [scripts/finetune_seq2seq.sh](scripts/finetune_seq2seq.sh). Change `DATA_ROOT, CHECKPOINT_PATH, SAVE_PATH` to your 
  local path. 
  
- Run the following script

```
bash scripts/finetune_seq2seq.sh \ 
     config_tasks/model_blocklm_large_generation.sh \ 
     config_tasks/seq_gigaword.sh
```
- For calculating rouge, install file2rouge from [here](https://github.com/pltrdy/files2rouge) and run `bash scripts/evaluate_seq2seq.sh`

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
* Download our [test set of wikibook](https://mailstsinghuaeducn-my.sharepoint.com/:t:/g/personal/duzx16_mails_tsinghua_edu_cn/EQa_B6KY_q1FjtUeG-T52iMBFtNrfhfHcZbzMxfkJKXKRQ?e=inTdHh) (or any dataset following the same format) and change `DATA_ROOT, CHECKPOINT_PATH` 
  in [scripts/evaluate_lm.sh](scripts/evaluate_lm.sh)
* Run the following script
  ```shell
  bash scripts/evaluate_lm.sh \ 
     config_tasks/model_blocklm_large_generation.sh \
     config_tasks/zero_lm.sh 
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