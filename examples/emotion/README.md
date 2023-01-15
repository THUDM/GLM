# Emotion

## Authors

**Armando Fortes**

Homepage: https://atfortes.github.io/

Contact: fmq22@mails.tsinghua.edu.cn

## Task Description

Emotion is a dataset of English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise. We follow the train-validation-test split configuration from [Huggingface](https://huggingface.co/datasets/emotion). Therefore, we use 16000 samples for training, 2000 samples for validation, and 2000 samples for testing. The goal of the task is: given an English Twitter message, classify whether it is shows sadness, joy, love, anger, fear, or surprise.

We perform prompt-based fine-tuning on the ```glm-roberta-large``` model and use prompt templates from [promptsource](https://github.com/bigscience-workshop/promptsource).

## Running Commands

You can run `python finetune.py --help` to see the usage of all the supported configurations. Using the default configuration as presented in the following command will reproduce the [reported results](#results). 

```bash
python finetune.py 
```

## Results

Using the above commands allows us to use the model version from best performing epoch on the validation set to test the performance on the test set. Accordingly, accuracy for ```glm-roberta-large``` on the ```emotion``` dataset increased from **25.85%** before fine-tuning to **93.35%** after fine-tuning, while the respective performance on the validation set was **94.45%**.

## Reference

```latex
@inproceedings{saravia-etal-2018-carer,
    title = "{CARER}: Contextualized Affect Representations for Emotion Recognition",
    author = "Saravia, Elvis  and
      Liu, Hsien-Chi Toby  and
      Huang, Yen-Hao  and
      Wu, Junlin  and
      Chen, Yi-Shin",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D18-1404",
    doi = "10.18653/v1/D18-1404",
    pages = "3687--3697",
    abstract = "Emotions are expressed in nuanced ways, which varies by collective or individual experiences, knowledge, and beliefs. Therefore, to understand emotion, as conveyed through text, a robust mechanism capable of capturing and modeling different linguistic nuances and phenomena is needed. We propose a semi-supervised, graph-based algorithm to produce rich structural descriptors which serve as the building blocks for constructing contextualized affect representations from text. The pattern-based representations are further enriched with word embeddings and evaluated through several emotion recognition tasks. Our experimental results demonstrate that the proposed method outperforms state-of-the-art techniques on emotion recognition tasks.",
}
```
