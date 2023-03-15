# GLM Squad Generation Dataset 

# Task Description 

+ **Dataset Name**: squad
+ **Authors**: Yu-Wen Michael Zhang, zywmichael2000@outlook.com, https://github.com/yuwenmichael.
+ **Task Description**: Generate answer given a context and question.
+ **Running Commands**: just run the .ipynb file. on GPU.
+ **Results**: By taking 1/38 of the training dataset (2308 instances) and evaluate on 1/38 of the validation dataset(278 instances), the result is the following (due to random sampling the train dataset, the score will be different if you run the .ipynb file (but not much) by yourself).
```
{'exact_match': 92.0863309352518, 'f1': 94.69410937415482}
```
+ **Reference**: 
The dataset is squad. Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
```
@article{2016arXiv160605250R,
       author = {{Rajpurkar}, Pranav and {Zhang}, Jian and {Lopyrev},
                 Konstantin and {Liang}, Percy},
        title = "{SQuAD: 100,000+ Questions for Machine Comprehension of Text}",
      journal = {arXiv e-prints},
         year = 2016,
          eid = {arXiv:1606.05250},
        pages = {arXiv:1606.05250},
archivePrefix = {arXiv},
       eprint = {1606.05250}
}
```

The comparison methods is F-1 score and exact match.

# Training Stage
## Save model
You must save your model in order to do evaluation !!!
Recommended path for storing the saved model is examples/squad/model_gen as the following (since there is already have all the config file you need except for `config.json` and `pytorch_model.bin`, which you need to run the code to generate it):
```
└── examples
    └── squad
        ├── README.md
        ├── requirements.txt
        ├── squad.ipynb
        └── model_gen
```
You need to first customise the path where you save the model in order to run the .ipynb file.

The best_model_path = '/home/zyw/squad/model_gen' is located in section GenerationTrainerClass.

Otherwise, the config.zip has all the configuration files you need. Just unzip it and put them in the same folder as the `pytorch_model.bin` and `config.json` and you are good to go.

## Hyperparameter
        train batch size = 4
        epoch = 1
        learning rate = 8e-6

## Sampling a small protion of the dataset
### setting
1. portion is used to sampling the dataset. when portion is 1, you are using the whole dataset. when portion is 38, you are using 1/38 of the dataset
2. random state is 1 when doing train test split to sample the dataset.


# Evaluation stage
### Load model with the path specified by best_model_path
In the provided code, the training set has 1/38 of the whole training dataset (shuffle = True, 2305 samples). The validation set has 1/38 of the whole testing dataset (shuffle = False, 278 samples). The model is trained on the training set and evaluated on the validation set.

The final result with the current setting is: 

{'exact_match': 92.0863309352518, 'f1': 94.69410937415482}

you can find this result in the end of the squad.ipynb as I have run the program myself.

# Contact
Should you have any problem, feel free to email me: zywmichael2000@outlook.com or Wechat ID: M_Zhang6