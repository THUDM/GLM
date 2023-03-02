# GLM

GLM is a General Language Model pretrained with an autoregressive blank-filling objective and can be finetuned on
various natural language understanding and generation tasks.

Please refer to our paper for a detailed description of GLM:

[GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/abs/2103.10360) (ACL
2022)

Zhengxiao Du*, Yujie Qian*, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, Jie Tang (*: equal contribution)

**We release [GLM-130B](https://github.com/THUDM/GLM-130B), an open bilingual (English & Chinese) pre-trained language
model wit 130 billion parameters based on the GLM framework.**

## Pretrained Models

You can download the pretrained models used in the paper
from [OneDrive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/zx-du20_mails_tsinghua_edu_cn/En6zA7_utRxHptKWZoDMO14Bkfj3uGRpslYkNvMPdGOmow?e=G0lGSc)
or [Tsinghua-Cloud](https://cloud.tsinghua.edu.cn/d/13f5b03da9594e5490c4).

| Name              | Params | Language | Corpus                                                                              | Objective      | File                                                               | Config                            |
|-------------------|--------|----------|-------------------------------------------------------------------------------------|----------------|--------------------------------------------------------------------|-----------------------------------|
| GLM-Base          | 110M   | English  | Wiki+Book                                                                           | Token          | glm-base-blank.tar.bz2                                             | model_blocklm_base.sh             |
| GLM-Large         | 335M   | English  | Wiki+Book                                                                           | Token          | glm-large-blank.tar.bz2                                            | model_blocklm_large.sh            |
| GLM-Large-Chinese | 335M   | Chinese  | [WuDaoCorpora](https://www.sciencedirect.com/science/article/pii/S2666651021000152) | Token+Sent+Doc | glm-large-chinese.tar.bz2                                          | model_blocklm_large_chinese.sh    |
| GLM-Doc           | 335M   | English  | Wiki+Book                                                                           | Token+Doc      | glm-large-generation.tar.bz2                                       | model_blocklm_large_generation.sh |
| GLM-410M          | 410M   | English  | Wiki+Book                                                                           | Token+Doc      | glm-1.25-generation.tar.bz2                                        | model_blocklm_1.25_generation.sh  |
| GLM-515M          | 515M   | English  | Wiki+Book                                                                           | Token+Doc      | glm-1.5-generation.tar.bz2                                         | model_blocklm_1.5_generation.sh   |
| GLM-RoBERTa       | 335M   | English  | RoBERTa                                                                             | Token          | glm-roberta-large-blank.tar.bz2                                    | model_blocklm_roberta_large.sh    |
| GLM-2B            | 2B     | English  | [Pile](https://arxiv.org/abs/2101.00027)                                            | Token+Sent+Doc | glm-2b.tar.bz2                                                     | model_blocklm_2B.sh               |
| GLM-10B           | 10B    | English  | [Pile](https://arxiv.org/abs/2101.00027)                                            | Token+Sent+Doc | [Download](https://lfs.aminer.cn/misc/cogview/glm-10b-1024.zip)    | model_blocklm_10B.sh              |
| GLM-10B-Chinese   | 10B    | Chinese  | [WuDaoCorpora](https://www.sciencedirect.com/science/article/pii/S2666651021000152) | Token+Sent+Doc | [Download](https://lfs.aminer.cn/misc/cogview/glm-10b-chinese.zip) | model_blocklm_10B_chinese.sh      |

Unzip the downloaded file into a local folder and set `CHECKPOINT_PATH` in the corresponding scripts to the folder path.

## Results

### [SuperGLUE](https://super.gluebenchmark.com)

dev set, single model, single-task finetuning

| Model                                                                                        | COPA | WSC  | RTE  | WiC  | CB        | MultiRC   | BoolQ | ReCoRD    |
|----------------------------------------------------------------------------------------------|------|------|------|------|-----------|-----------|-------|-----------|
| GLM-10B                                                                                      | 98.0 | 95.2 | 93.1 | 75.7 | 98.7/98.2 | 88.1/63.3 | 88.7  | 94.4/94.0 |
| [DeBERTa-XXLarge-v2](https://github.com/microsoft/DeBERTa/tree/master/experiments/superglue) | 97.0 | -    | 93.5 | -    | -         | 87.8/63.6 | 88.3  | 94.1/93.7 |

### Seq2Seq

[CNN/Daily Mail](https://github.com/abisee/cnn-dailymail) (test set, no additional data used)

| Model         | ROUGE-1  | ROUGE-2  | ROUGE-L  |
|---------------|----------|----------|----------|
| GLM-10B       | **44.7** | 21.4     | **41.4** |
| T5-11B        | 43.5     | **21.6** | 40.7     |
| PEGASUS-Large | 44.2     | 21.5     | **41.4** |
| BART-Large    | 44.2     | 21.3     | 40.9     |

[XSum](https://github.com/EdinburghNLP/XSum) (test set, no additional data used)

| Model         | ROUGE-1  | ROUGE-2  | ROUGE-L  |
|---------------|----------|----------|----------|
| GLM-10B       | **48.9** | **25.7** | **40.4** |
| PEGASUS-Large | 47.2     | 24.6     | 39.3     |
| BART-Large    | 45.1     | 22.3     | 37.3     |

### Language Modeling

test set, zero-shot

| Model              | LAMBADA (accuracy) | Wikitext103 (perplexity) |
|--------------------|--------------------|--------------------------|
| GLM-10B (bi)       | 72.35              | 11.33                    |
| GLM-10B (uni)      | 67.18              | 12.22                    |
| GPT-2              | 52.66              | 17.48                    |
| Megatron-LM (8.3B) | 66.51              | 10.81                    |
| Turing-NLG         | 67.98              | 10.21                    |

## Get Started

### Hugging Face Hub

You can access GLM models via HuggingFace Hub. Please
install `transformers>=4.23.1` and find all the available models [here](https://huggingface.co/models?filter=glm,thudm).

#### Generation
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-10b", trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained("THUDM/glm-10b", trust_remote_code=True)
model = model.half().cuda()
model.eval()

# Inference
inputs = tokenizer("Ng is an adjunct professor at [MASK] (formerly associate professor and Director of its Stanford AI Lab or SAIL ). Also a pioneer in online education, Ng co-founded Coursera and deeplearning.ai.", return_tensors="pt")
inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
inputs = inputs.to('cuda')
outputs = model.generate(**inputs, max_length=512, eos_token_id=tokenizer.eop_token_id)
print(tokenizer.decode(outputs[0].tolist()))

# Training
inputs = tokenizer(
    ["Tsinghua University is located in [MASK].", "One minus one equals zero, is it correct? Answer: [MASK]"],
    return_tensors="pt", padding=True)
inputs = tokenizer.build_inputs_for_generation(inputs, targets=["Beijing", "No"], max_gen_length=8, padding=False)
inputs = inputs.to('cuda')
outputs = model(**inputs)
loss = outputs.loss
logits = outputs.logits
```
#### Classification
```python
from transformers import AutoTokenizer, AutoModelForMultipleChoice
tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-10b", trust_remote_code=True)
model = AutoModelForMultipleChoice.from_pretrained("THUDM/glm-10b", trust_remote_code=True)
model = model.half().cuda()
model.eval()

inputs = tokenizer(["Tsinghua University is located in [MASK].",
                    "One minus one equals zero, is it correct? Answer: [MASK]"], return_tensors="pt", padding=True)
choices = [["Beijing", "Shanghai"], ["Yes", "No"]]
inputs = tokenizer.build_inputs_for_multiple_choice(inputs, choices)
inputs = inputs.to('cuda')
outputs = model(**inputs)
logits = outputs.logits
```
You can also convert the finetuned checkpoints with `scripts/convert_glm_checkpoint_to_transformers.py`. 
### Docker Image

We prepare two docker images based on CUDA 10.2 and CUDA 11.2. You can pull the pre-built images from Docker Hub and run
with docker v19.03+

  ```shell
  docker run --gpus all --rm -it --ipc=host zxdu20/glm-cuda102
  ```

or replace `glm-cuda102` with `glm-cuda112`.

You can also modify the image according to your requirements in [docker/cuda102.dockerfile](docker/cuda102.dockerfile)
and build the image yourself

  ```shell
    docker build -f cuda102.dockerfile . -t glm-cuda102
  ```

### Manual Installation

Please first install PyTorch (we use 1.7.0) and [apex](https://github.com/NVIDIA/apex), and then install other
dependencies by `pip install -r requirements.txt`

### Clone this repo

  ```shell
  git clone https://github.com/THUDM/GLM
  cd GLM
  ```

### Model Parallelism

If your encounter the `CUDA out of memory` error, which means you GPU memory is limited, you can try the model
parallelism to divide the parameters into multiple GPUs. Take the two-way model parallelism as an example. First
run `change_mp.py` to divide the checkpoint:

```shell
python change_mp.py path_to_the_checkpoint 2
```

Then update the checkpoint path in the model config file (such
as [config_tasks/model_blocklm_10B.sh](config_tasks/model_blocklm_10B.sh)) and change `MP_SIZE` in the script (such
as [scripts/ds_finetune_superglue.sh](scripts/ds_finetune_superglue.sh)) to `2`.

## Usage

We provide scripts for finetuning GLM on some downstream tasks.

### Left-to-Right Generation / Blank Filling (Interactive)

* Change `CHECKPOINT_PATH` to your local path. Run the following script

```
bash scripts/generate_block.sh \
     config_tasks/model_blocklm_10B_chinese.sh
```

Some models (GLM-2B, GLM-10B, and GLM-10B-Chinese) use three different mask tokens: `[MASK]` for short blank
filling, `[sMASK]` for sentence filling, and `[gMASK]` for left-to-right generation.

<details>
<summary><b>Examples</b></summary>

#### Usage of `[MASK]` (Entity Prediction):

##### Example1

Context: Ng is an adjunct professor at [MASK] (formerly associate professor and Director of its Stanford AI Lab or SAIL ). Also a pioneer in online education, Ng co-founded Coursera and deeplearning.ai.

GLM: the stanford university

##### Example2 (Chinese)

Context: 凯旋门位于意大利米兰市古城堡旁。1807年为纪念[MASK]而建，门高25米，顶上矗立两武士青铜古兵车铸像。

GLM:拿破仑军队攻克米兰城

#### Usage of `[sMASK]` (Sentence Prediction)

##### Example3

Context: There have been various types of pretraining architectures including autoencoding models (e.g., BERT),
autoregressive models (e.g., GPT), and encoder-decoder models (e.g., T5). [sMASK] We propose a General Language Model (
GLM) based on autoregressive blank infilling to address this challenge. GLM improves blank filling pretraining by adding
2D positional encodings and allowing an arbitrary order to predict spans, which results in performance gains over BERT
and T5 on NLU tasks. Meanwhile, GLM can be pretrained for different types of tasks by varying the number and lengths of
blanks. On a wide range of tasks across NLU, conditional and unconditional generation, GLM outperforms BERT, T5, and GPT
given the same model sizes and data, and achieves the best performance from a single pretrained model with 1.25×
parameters of BERT Large, demonstrating its generalizability to different downstream tasks.

GLM: However, there is a growing need to develop a single pretraining model that is not only good at natural language
understanding (NLU) or dialog generation/generation (dialog), but is also able to predict other tasks such as sentiment
analysis, conditional generation, or machine translation (MT).

##### Example4 (Chinese)

Context: 工业互联网（Industrial
Internet）是新一代信息通信技术与工业经济深度融合的新型基础设施、应用模式和工业生态，通过对人、机、物、系统等的全面连接，构建起覆盖全产业链、全价值链的全新制造和服务体系，为工业乃至产业数字化、网络化、智能化发展提供了实现途径，是第四次工业革命的重要基石。[sMASK]
它以网络为基础、平台为中枢、数据为要素、安全为保障，既是工业数字化、网络化、智能化转型的基础设施，也是互联网、大数据、人工智能与实体经济深度融合的应用模式，同时也是一种新业态、新产业，将重塑企业形态、供应链和产业链。当前，工业互联网融合应用向国民经济重点行业广泛拓展，形成平台化设计、智能化制造、网络化协同、个性化定制、服务化延伸、数字化管理六大新模式，赋能、赋智、赋值作用不断显现，有力的促进了实体经济提质、增效、降本、绿色、安全发展。

GLM: 工业互联网是制造业技术、管理、模式的重大变革,是推动互联网、大数据、人工智能和实体经济深度融合的重要载体,是建设制造强国和网络强国的重要基础。

#### Usage of `[gMASK]` (Long Text Generation)

##### Example5 (Chinese)

Context: 问题：冬天，中国哪座城市最适合避寒？问题描述：能推荐一些国内适合冬天避寒的城市吗？回答用户：旅游爱好者 回答： [gMASK]

GLM:
海南三亚,三亚的冬天是三亚的旺季,因为冬天国内大部分地方都有雾霾,而三亚却是没有雾霾的地方,所以三亚是冬天最惬意不过的地方了。在东北长大的我觉得三亚简直就是一个天堂,冬天去海口三亚旅游,享受冬天,享受阳光沙滩。但是海口却是很干燥,需要多喝水。
三亚冬天最热门的玩法就是晒太阳,在海边晒晒太阳,在沙滩上晒晒太阳,感觉整个人都得到了解放。三亚还有一个特色项目,就是海上冲浪,在三亚的沙滩上冲浪也是一件非常刺激的事情。
海口,海口冬季的阳光十分温暖,海南的冬季也是属于冬季旅游的旺季。冬季的海口最棒的是去海南的热带野生动植物园,那里有数之不尽的热带小动物,在这里可以近距离的和它们接触,海南的热带野生动植物园也是海南的天然氧吧。还可以在海口观澜湖公园里感受海口美丽的海景。
贵阳,贵州的冬天也是十分温暖的,贵阳也是冬季避寒很好的城市之一。冬季去贵阳玩一定要去黔灵山,黔灵山是贵州香火很旺盛的一个寺庙,寺庙的冬季香火鼎盛,在冬季去寺庙游玩也是一个很好的体验。除了黔灵山,贵阳在冬季还有花溪公园可以去玩,花溪公园也是去当地公园玩最好的选择。
青岛,青岛的冬天是青岛最舒服的时候,青岛有很多海滨浴场,冬天去海边泡一泡温泉,然后晒晒太阳是一件十分惬意的事情。青岛也有沙滩,冬天在沙滩上晒晒太阳,看看海,再玩玩沙滩游戏,感觉十分快乐的事。
</details>

You can also add multiple `[MASK]` and `[sMASK]` in a single example. The model will fill the blanks one by one from left to right. The answer to each blank always begins with a special `<|startofpiece|>`.

<details>
<summary><b>Examples</b></summary>

##### Example1

Context: There have been various types of pretraining architectures including autoencoding models (e.g., BERT), autoregressive models (e.g., GPT), and [MASK] (e.g., T5). [sMASK] We propose a General Language Model ( GLM) based on autoregressive blank infilling to address this challenge. GLM improves blank filling pretraining by adding 2D positional encodings and allowing an arbitrary order to predict spans, which results in performance gains over [MASK] on NLU tasks. Meanwhile, GLM can be pretrained for different types of tasks by varying the number and lengths of blanks. On a wide range of tasks across NLU, conditional and [MASK], GLM outperforms BERT, T5, and GPT given the same model sizes and data, and achieves the best performance from a single pretrained model with 1.25× parameters of BERT Large , demonstrating its generalizability to different downstream tasks.

GLM: <|startofpiece|> blank filling models<|startofpiece|> However, most of them cannot easily transfer to other downstream tasks due to the different characteristics of these tasks.<|startofpiece|> other pretrained models<|startofpiece|> unconditional reading, and semantic role labeling tasks

##### Example2 (Chinese)

Context: 工业互联网（Industrial Internet）是新一代[MASK]与[MASK]深度融合的新型基础设施、应用模式和工业生态，通过对人、机、物、系统等的全面连接，构建起覆盖全产业链、全价值链的全新制造和服务体系，为工业乃至产业数字化、网络化、智能化发展提供了实现途径，是第四次工业革命的重要基石。[sMASK] 它以网络为基础、平台为中枢、数据为要素、安全为保障，既是工业数字化、网络化、智能化转型的基础设施，也是互联网、大数据、人工智能与实体经济深度融合的应用模式，同时也是一种新业态、新产业，将重塑企业形态、供应链和产业链。当前，工业互联网融合应用向国民经济重点行业广泛拓展，形成[MASK]、智能化制造、[MASK]、个性化定制、服务化延伸、数字化管理六大新模式，赋能、赋智、赋值作用不断显现，有力的促进了实体经济提质、增效、降本、绿色、安全发展。

GLM:
<|startofpiece|>信息技术(ICT)<|startofpiece|>工业经济(II2O)<|startofpiece|>我国工业互联网是面向工业全领域、全流程、全体系的互联网,具有多产业、多领域融合的特点。<|startofpiece|>网络化协同<|startofpiece|>平台企业

</details>


### SuperGLUE

- Download the [SuperGlue](https://super.gluebenchmark.com/tasks) data and check the experiment setup in
  [scripts/ds_finetune_superglue.sh](scripts/ds_finetune_superglue.sh). Note
  that `DATA_ROOT, CHECKPOINT_PATH, SAVE_PATH`
  need to be changed to your local path. You may also change the `batch-size` and `nproc_per_node` according to your
  available hardware.

- Run the following script (use the COPA dataset as an example)

```
bash scripts/ds_finetune_superglue.sh \
     config_tasks/model_blocklm_10B.sh \
     config_tasks/task_copa.sh
```

- We also implement [P-Tuning](https://arxiv.org/abs/2103.10385) in our code. Run the following script to integrate
  p-tuning:

```shell
bash scripts/ds_finetune_superglue_prompt.sh \
     config_tasks/model_blocklm_10B.sh \
     config_tasks/task_copa.sh
```

- To apply GLM to a new NLU dataset with cloze-filling finetuning, implement a `DataProcessor` in
  [tasks/superglue/dataset.py](tasks/superglue/dataset.py) for data loading and add a `PVP` in
  [tasks/superglue/pvp.py](tasks/superglue/pvp.py) for the cloze question. More details can be found
  [here](tasks/superglue/README.md).

### Seq2Seq

- Download the [Gigaword](https://github.com/harvardnlp/sent-summary)
  , [CNN/Daily Mail](https://github.com/artmatsak/cnn-dailymail)
  or [XSum](https://github.com/EdinburghNLP/XSum/tree/master/XSum-Dataset) dataset and check the experiment setup in
  [scripts/ds_finetune_seq2seq.sh](scripts/ds_finetune_seq2seq.sh). Change `DATA_ROOT, CHECKPOINT_PATH, SAVE_PATH` to
  your
  local path.

- Run the following script (use the CNN/Daily Mail dataset as an example)

  ```
  bash scripts/ds_finetune_seq2seq.sh \ 
     config_tasks/model_blocklm_10B.sh \ 
     config_tasks/seq_cnndm_org.sh
  ```
- The summaries are written into `./runs/experiment_name/test.jsonl.hyps`. The references are written
  into `test.jsonl.refs` in the same directory. For calculating rouge,
  install [file2rouge](https://github.com/pltrdy/files2rouge) and download Stanford CoreNLP
  from [here](http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip). Run the following script
  ```
  bash scripts/evaluate_seq2seq.sh \
   ./runs/experiment_name/test.jsonl.hyps ./runs/experiment_name/test.jsonl.refs
  ```

#### Train with your own data

Process your seq2seq data into `{split}.source` and `{split}.target`, with each line being the context or the target of
a sample, and `split` being `train`, `val`, and `test`.

Run the following script

```shell
bash scripts/ds_finetune_seq2seq.sh \ 
   config_tasks/model_blocklm_10B.sh \ 
   config_tasks/seq_customization.sh
```

You can specify the hyperparameters in `config_tasks/seq_customization.sh`
and `config_tasks/config_blocklm_10B_cnndm.json`

### Multiple Choice (Zero-shot)

```shell
bash scripts/evaluate_multichoice.sh config_tasks/model_blocklm_10B.sh
```

Note that `CHECKPOINT_PATH` and `DATA_PATH` need to be changed to your local path.

The format of each line of the data file should be

```
{"inputs_pretokenized": "Context and question here", "choices_pretokenized": ["Choice 1", "Choice 2", "Choice 3"], "label": int}
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

* Download
  our [test set of wikibook](https://mailstsinghuaeducn-my.sharepoint.com/:t:/g/personal/duzx16_mails_tsinghua_edu_cn/EQa_B6KY_q1FjtUeG-T52iMBFtNrfhfHcZbzMxfkJKXKRQ?e=inTdHh)
  or [Wikitext103](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip) dataset and
  change `DATA_ROOT, CHECKPOINT_PATH`
  in [scripts/evaluate_lm.sh](scripts/evaluate_lm.sh)
* Run the following script
  ```shell
  bash scripts/evaluate_lm.sh \ 
     config_tasks/model_blocklm_large_generation.sh \
     config_tasks/zero_wikitext.sh 
  ```

### Text Infilling

- Download the [Yahoo](https://github.com/Varal7/blank_language_model) dataset and check the experiment setup in
  [scripts/finetune_blank.sh](scripts/finetune_blank.sh). Change `DATA_ROOT, CHECKPOINT_PATH, SAVE_PATH` to your
  local path.

- Run the following script

```
bash scripts/finetune_blank.sh \ 
     config_tasks/model_blocklm_large.sh \ 
     config_tasks/seq_blank.sh
```

## Pretrain

Run the following script to pre-train the GLM-Large model

```shell
bash scripts/ds_pretrain_nvidia.sh config/ds_block_large.sh
```

The script [scripts/ds_pretrain_nvidia.sh](scripts/ds_pretrain_nvidia.sh) launches the training program with DeepSpeed.
You should change `NUM_WORKERS` and `NUM_GPUS_PER_WORKER` to the number of workers and the number of gpus per worker.
Also change `HOST_FILE_PATH` to the path to an OpenMPI-style hostfile. More details about DeepSpeed launcher can be
found [here](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node).

The file [config/ds_block_large.sh](config/ds_block_large.sh) defines the hyperparameters for pretraining. Most of the
arguments are fairly self-explanatory. Specifically, `--train-data` can be multiple keywords defined in `NAMED_CORPORA`
in [data_utils/corpora.py](data_utils/corpora.py). The hyperparameters of the optimizer are defined in the corresponding
json file under `config`. The semantics of the json file can be found [here](https://www.deepspeed.ai/docs/config-json).

## Citation

Part of the code is based on [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
and [PET](https://github.com/timoschick/pet).

Please cite our paper if you find this code useful for your research:

```
@article{DBLP:conf/acl/DuQLDQY022,
  author    = {Zhengxiao Du and
               Yujie Qian and
               Xiao Liu and
               Ming Ding and
               Jiezhong Qiu and
               Zhilin Yang and
               Jie Tang},
  title     = {{GLM:} General Language Model Pretraining with Autoregressive Blank Infilling},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational
               Linguistics (Volume 1: Long Papers), {ACL} 2022, Dublin, Ireland,
               May 22-27, 2022},
  pages     = {320--335},
  publisher = {Association for Computational Linguistics},
  year      = {2022},
}
```