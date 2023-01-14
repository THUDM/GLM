# Dataset Name: [art](https://huggingface.co/datasets/art)

# Authors: 

Jember Liku Amare (lik22@mails.tsinghua.edu.cn) <br>
Student ID: 2022280073

# Task Description

'art' is a dataset consisting of observation and hypothesis pairs. Each entry of this dataset consists of two observations and two hypotheses. The task is to choose the best hypothesis based on the given observations. There is a label column in the dataset. The value on this column for each entry is either 1 or 2, refering to hypothesis 1 or hypothesis 2. The dataset has:
<ul>
    <li> 169,654 train and 
    <li> 1,532 validation data.
</ul>

# Running Commands: 

Running the cells in the given notebook gives the desired results.

Hyperparameters:
<ul>
    <li> epoch = 2
    <li> batch_size = 8
    <li> learning_rate = variable with scheduler
</ul>

# Results: 
To evaluate the model's performance, the validation was done both before and after fine tuning. These are the results for the whole process.
<ul>
    <li> Accuracy before fine tuning = 0.503
    <li> Accuracy after fine tuning = 0.733
    <li> Training Loss = 0.004
    <li> Training Accuracy = 0.969
    <li> Validation Loss = 0.873
    <li> Validation Accuracy = 0.731
</ul>

# Reference: 
```
@inproceedings {Bhagavatula2020Abductive,
    title = {Abductive Commonsense Reasoning},
    author = {Chandra Bhagavatula and Ronan Le Bras and Chaitanya Malaviya and Keisuke Sakaguchi and Ari Holtzman and Hannah Rashkin and Doug Downey and Wen-tau Yih and Yejin Choi},
    booktitle = {International Conference on Learning Representations},
    year = {2020},
    url = {https://openreview.net/forum?id=Byg1v1HKDB}
}
```