# Dataset Name as Title

Author: Jingcheng Hu, hujc22@mails.tsinghua.edu.cn, https://reign12.github.io/

Student ID: 2022312848

## Task Description
### Dataset Statistics
We are following the train-validation-test split from [Huggingface multi_news](https://huggingface.co/datasets/multi_news).
The train-validation-test split are: training (80\%, 44,972), validation (10\%, 5,622), and test (10\%, 5,622) sets.
Note that in multi_news, the summaries are notably longer than in other works, about 260 words on average(the original texts are obiviously even longer), posing hard challenges for GLM-RoBERTa-Large or other similar models whose maximum length is 512, which makes it extremely difficult to cover all points in the original text while generating sufficiently long summary.

### Task Introduction
For task prompt, we are using prompt templates from [promptsource](https://github.com/bigscience-workshop/promptsource)
The task of multi_news is summarization task from multiple documents. Here is an example:
```python
{
    "document": "some line val \n another line",
    "summary": "target val line"
}
```

## How to Train and Eval
### Dependency
You can activate your own conda env and run command
```bash
bash env_setup.sh cuda # If you are running on nvidia GPUs

bash env_setup.sh rocm # If you are running on amd GPUs
```
### Training and Evaluation
You can run `python main.py --help` or directly go to `./config/basic.yaml` to see all the supported configuration.

To run the distributed training, which will evaluate the results along the way; per step loss, per epoch loss and final Rouge scores will be recorded:
```bash
torchrun --nproc_per_node <YOUR_GPU_NUM> main.py \
    task="pg" \ # this is a Prompted Genertation task
    data.dataset="multi_news" \
    model.name="BAAI/glm-roberta-large" \ # we also support bert-large-uncased, roberta-large
    data.prompt_id="0" \ # prompt_id of original_task=True prompt templates from promptsource; for the name of each prompt, you can refer to training log as you start the job, which will be like "train dataset prompt_key  ['distill', 'summarize', 'summary scenario', 'synthesize', 'what are the key points']"
    jobname=<ANY_NAME_YOU_LIKE> \
    debug=False \ # If you want to disable wandb, set debug=True; you can setup your wandb related var as env var, or just type it when the program need it; refer to logger.py for details
    optimizer.lr="5r-5" \ # no lr scaling will be done, this lr will be the final lr
    trainer.batch="8" \ # this is the total batch summed in all cards
    trainer.accumulate_steps="4" \ # we support gradient accumulate steps to have larger effective batch size
    trainer.warmup_start="1e-8" \ # warmup_lr at start
    trainer.epochs="3" trainer.warmup_epochs="1" # we use linear warmup and cosine decay
    # there are some more configs can be changed, please refer to ./config/basic.yaml for details and simply follow the pattern here
```

## Results
We use Rouge-1-{P,R} as the metrics
The final epoch Rouge-1-{P,R} using above commands are 45.19,22.95 on test set.
For T5-Large, we tuning the learning rate and the best performances are 46.15,20.56. 

|Model|Rouge-1-P | Rouge-1-R | 
|:---:|:---:|
|glm-roberta-large | 45.19|22.95 
|t5-large |46.15|20.56 

## Reference
multi_news dataset paper:
```bibtex
@misc{fabbriMultiNewsLargeScaleMultiDocument2019,
  title = {Multi-{{News}}: A {{Large-Scale Multi-Document Summarization Dataset}} and {{Abstractive Hierarchical Model}}},
  shorttitle = {Multi-{{News}}},
  author = {Fabbri, Alexander R. and Li, Irene and She, Tianwei and Li, Suyi and Radev, Dragomir R.},
  date = {2019-06-19},
  number = {arXiv:1906.01749},
  eprint = {1906.01749},
  eprinttype = {arxiv},
  primaryclass = {cs},
  publisher = {{arXiv}},
  doi = {10.48550/arXiv.1906.01749},
  url = {http://arxiv.org/abs/1906.01749},
  urldate = {2023-01-02},
  archiveprefix = {arXiv}
}

```
For T5 we are following the huggingface implementation, and the original paper is:
```bibtex
@article{raffelExploringLimitsTransfer,
  title = {Exploring the {{Limits}} of {{Transfer Learning}} with a {{Uniﬁed Text-to-Text Transformer}}},
  author = {Raffel, Colin and Shazeer, Noam and Roberts, Adam and Lee, Katherine and Narang, Sharan and Matena, Michael and Zhou, Yanqi and Li, Wei and Liu, Peter J},
  pages = {67},
  langid = {english}
}
```