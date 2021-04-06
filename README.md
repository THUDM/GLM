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

- Download the [SuperGlue](https://super.gluebenchmark.com/tasks) data and change `DATA_ROOT` and `CHECKPOINT_PATH` in 
  [scripts/finetune_superglue.sh](scripts/finetune_superglue.sh). Also change `batch-size` and `nproc_per_node` according to your available hardware. To reproduce our results, we use aggregated batch size 32 for `MultiRC` and `ReCORD` and 16 for other tasks.  
- Run the following script (use the COPA dataset as an example)

```
bash scripts/finetune_superglue.sh \
     config_tasks/model_blocklm_roberta_large.sh \
     config_tasks/task_copa.sh
```

- To apply GLM to a new NLU dataset with cloze-filling finetuning, add a `DataProcessor` in 
  [tasks/superglue/dataset.py](tasks/superglue/dataset.py) for data loading and add a `PVP` in 
  [tasks/superglue/pvp.py](tasks/superglue/pvp.py) for the cloze question. More details can be found [here](https://github.com/timoschick/pet#-train-your-own-pet).

### Text Summarization

- Download the [Gigaword](https://github.com/harvardnlp/sent-summary) data and change `DATA_ROOT` and `CHECKPOINT_PATH` 
  in [scripts/finetune_seq2seq.sh](scripts/finetune_seq2seq.sh)
  
- Run the following script

```
bash scripts/finetune_seq2seq.sh \ 
     config_tasks/model_blocklm_large_generation.sh \ 
     config_tasks/seq_gigaword.sh
```
### Language Modeling
#### LAMBADA Cloze Accuracy
* Download the [LAMBADA](https://github.com/cybertronai/bflm/blob/master/lambada_test.jsonl) data and change `DATA_ROOT` and `CHECKPOINT_PATH` in [scripts/evaluate_lm.sh](scripts/evaluate_lm.sh)
* Run the following script
```shell
bash scripts/evaluate_lm.sh \ 
     config_tasks/model_blocklm_large_generation.sh \
     config_tasks/zero_lambada.sh 
```
#### LM Perplexity
* Download our test set of wikibook (or any dataset following the same format) and hange `DATA_ROOT` and `CHECKPOINT_PATH` in [scripts/evaluate_lm.sh](scripts/evaluate_lm.sh)
* Run the following script
```shell
bash scripts/evaluate_lm.sh \ 
     config_tasks/model_blocklm_large_generation.sh \
     config_tasks/zero_lm.sh 