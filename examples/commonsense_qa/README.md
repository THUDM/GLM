# commonsense_qa

Author: Jingcheng Hu, hujc22@mails.tsinghua.edu.cn, https://reign12.github.io/

Student ID: 2022312848

## Task Description
### Dataset Statistics
We are following the train-validation-test split from [Huggingface commonsense_qa](https://huggingface.co/datasets/commonsense_qa).
However, note that for test set the answers are not provided, so we are not using test set in our experiments. There are 9741 samples in training set and 1221 samples in validation set.

### Task Introduction
For task prompt, we are using prompt templates from [promptsource](https://github.com/bigscience-workshop/promptsource).
The task of commonsense_qa is a multiple-choice question answering challenge requiring rich world knowledges. 
Here is an example:
```python
{
    'id': '075e483d21c29a511267ef62bedc0461',
    'question': 'The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?',
    'question_concept': 'punishing',
    'choices': {'label': ['A', 'B', 'C', 'D', 'E'],
    'text': ['ignore', 'enforce', 'authoritarian', 'yell at', 'avoid']},
    'answerKey': 'A'
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

To run the distributed training, which will evaluate the results along the way; per step loss, per epoch loss and per epoch accuracy will be recorded:
```bash
torchrun --nproc_per_node <YOUR_GPU_NUM> main.py \
    task="pc" \ # this is a Prompted Choice task
    data.dataset="commonsense_qa" \
    model.name="BAAI/glm-roberta-large" \ # we also support bert-large-uncased, roberta-large
    data.prompt_id="2" \ # prompt_id of original_task=True prompt templates from promptsource; for the name of each prompt, you can refer to training log as you start the job, which will be like "train dataset prompt_key  ['answer_given_question_without_options', 'most_suitable_answer', 'question_answering', 'question_to_answer_index']"
    jobname=<ANY_NAME_YOU_LIKE> \
    debug=False \ # If you want to disable wandb, set debug=True; you can setup your wandb related var as env var, or just type it when the program need it; refer to logger.py for details
    optimizer.lr="1r-5" \ # no lr scaling will be done, this lr will be the final lr
    trainer.batch="32" \ # this is the total batch summed in all cards
    trainer.accumulate_steps="2" \ # we support gradient accumulate steps to have larger effective batch size
    trainer.epochs="10" trainer.warmup_epochs="1" # we use linear warmup and cosine decay
    # there are some more configs can be changed, please refer to ./config/basic.yaml for details and simply follow the pattern here
```

## Results
Report final performance and other methods.

The final epoch accuracy using above commands are 64.22% on validation set. 
For RoBERTa-Large and BERT-Large, we tuning the learning rate and the best performances are 74.59% and 62.90% for each.

| Model | Accuracy |
|:---:|:---:|
glm-roberta-large | 64.22 |
roberta-large | 74.59 |
bert-large-uncased | 62.90 |

## Reference
### Dataset
commonsense_qa dataset paper
```latex
@inproceedings{Talmor2019,
  title = {{{CommonsenseQA}}: {{A Question Answering Challenge Targeting Commonsense Knowledge}}},
  shorttitle = {{{CommonsenseQA}}},
  booktitle = {Proceedings of the 2019 {{Conference}} of the {{North American Chapter}} of the {{Association}} for {{Computational Linguistics}}: {{Human Language Technologies}}, {{Volume}} 1 ({{Long}} and {{Short Papers}})},
  author = {Talmor, Alon and Herzig, Jonathan and Lourie, Nicholas and Berant, Jonathan},
  date = {2019-06},
  pages = {4149--4158},
  publisher = {{Association for Computational Linguistics}},
  location = {{Minneapolis, Minnesota}},
  doi = {10.18653/v1/N19-1421},
  url = {https://aclanthology.org/N19-1421},
  urldate = {2023-01-07},
  eventtitle = {{{NAACL-HLT}} 2019}
}
```
commonsense_qa huggingface link are mentioned in above.

roberta and bert we are directly using huggingface implementation. The original papers are:
```latex
@misc{Liu2019,
  title = {{{RoBERTa}}: {{A Robustly Optimized BERT Pretraining Approach}}},
  shorttitle = {{{RoBERTa}}},
  author = {Liu, Yinhan and Ott, Myle and Goyal, Naman and Du, Jingfei and Joshi, Mandar and Chen, Danqi and Levy, Omer and Lewis, Mike and Zettlemoyer, Luke and Stoyanov, Veselin},
  date = {2019-07-26},
  number = {arXiv:1907.11692},
  eprint = {1907.11692},
  eprinttype = {arxiv},
  primaryclass = {cs},
  publisher = {{arXiv}},
  url = {http://arxiv.org/abs/1907.11692},
  urldate = {2023-01-07},
  archiveprefix = {arXiv},
  version = {1}
}

@unpublished{devlinBERTPretrainingDeep2019,
  title = {{{BERT}}: {{Pre-training}} of {{Deep Bidirectional Transformers}} for {{Language Understanding}}},
  shorttitle = {{{BERT}}},
  author = {Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  date = {2019-05-24},
  eprint = {1810.04805},
  eprinttype = {arxiv},
  primaryclass = {cs},
  url = {http://arxiv.org/abs/1810.04805},
  urldate = {2022-04-12},
  archiveprefix = {arXiv}
}
```