# GLM

GLM is a General Language Model pretrained with an autoregressive blank-filling objective and can be finetuned on 
various natural language understanding and generation tasks. 

Please refer to our paper for a detailed description of GLM:

[All NLP Tasks Are Generation Tasks: A General Pretraining Framework](https://arxiv.org/abs/2103.10360)

Zhengxiao Du*, Yujie Qian*, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, Jie Tang (*: equal contribution)

## Pretrained Models

- [GLM-base]()
- [GLM-large]()
- [GLM-roberta]()


## Usage
We provide scripts for finetuning GLM on some downstream tasks.

### SuperGLUE

- Download the [SuperGlue](https://super.gluebenchmark.com/tasks) data and check the experiment setup in 
  [scripts/finetune_superglue.sh](scripts/finetune_superglue.sh). Note that `DATA_ROOT, CHECKPOINT_PATH, SAVE_PATH` 
  need to be changed to your local path.

- Run the following script (use the COPA dataset as an example)

```
bash scripts/finetune_superglue.sh \
     config_tasks/model_blocklm_roberta_large.sh \
     config_tasks/task_copa.sh
```

- To apply GLM to a new NLU dataset with cloze-filling finetuning, add a `DataProcessor` in 
  [tasks/superglue/dataset.py](tasks/superglue/dataset.py) for dataset loading and add a `PVP` in 
  [tasks/superglue/pvp.py](tasks/superglue/pvp.py) for the cloze question. Our implementation is inherited from PET and
  more details can be found [here](https://github.com/timoschick/pet#-train-your-own-pet).

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
