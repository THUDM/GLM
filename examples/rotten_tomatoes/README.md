# Rotten Tomatoes

## Authors

**Armando Fortes**

Homepage: https://atfortes.github.io/

Contact: fmq22@mails.tsinghua.edu.cn

## Task Description

Movie Review Dataset. This dataset contains 5,331 positive and 5,331 negative processed sentences from Rotten Tomatoes movie reviews. We follow the train-validation-test split configuration from [HuggingFace](https://huggingface.co/datasets/rotten_tomatoes). Therefore, we use 8530 samples for training, 1066 samples for validation, and 1066 samples for testing. The goal of the task is: given a Rotten Tomatoes movie review, classify whether it is positive or negative.

We perform prompt-based fine-tuning on the ```glm-roberta-large``` model and use prompt templates from [promptsource](https://github.com/bigscience-workshop/promptsource).

## Running Commands

You can run `python finetune.py --help` to see the usage of all the supported configurations. Using the default configuration as presented in the following command will reproduce the [reported results](#results). 

```bash
python finetune.py 
```

## Results

Using the above commands allows us to use the model version from best performing epoch on the validation set to test the performance on the test set. Accordingly, accuracy for ```glm-roberta-large``` on the ```rotten_tomatoes``` dataset increased from **50.75%** before fine-tuning to **88.93%** after fine-tuning, while the respective performance on the validation set was **90.24%**.

## Reference

```latex
@InProceedings{Pang+Lee:05a,
  author =       {Bo Pang and Lillian Lee},
  title =        {Seeing stars: Exploiting class relationships for sentiment
                  categorization with respect to rating scales},
  booktitle =    {Proceedings of the ACL},
  year =         2005
}
```
