# GLM

GLM 是用自回归完形填空预训练的一个通用语言模型，可以在各种自然语言理解和生成任务中进行微调。

有关GLM的详细描述，请参阅我们的论文：

[GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/abs/2103.10360) (ACL 2022)

Zhengxiao Du*, Yujie Qian*, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, Jie Tang (*: equal contribution)

**好消息: 我们发布了具有60亿参数的一个开源预训练语言模型 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B), 该模型基于GLM框架对中文QA和对话进行了优化。**

[//]: # (**We release [GLM-130B]&#40;https://github.com/THUDM/GLM-130B&#41;, an open bilingual &#40;English & Chinese&#41; pre-trained language model with 130 billion parameters based on the GLM framework.**)

## 预训练模型

你可以从[OneDrive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/zx-du20_mails_tsinghua_edu_cn/En6zA7_utRxHptKWZoDMO14Bkfj3uGRpslYkNvMPdGOmow?e=G0lGSc)
或[Tsinghua-Cloud](https://cloud.tsinghua.edu.cn/d/13f5b03da9594e5490c4)下载论文中使用的预训练模型。

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


将下载的文件解压到本地文件夹中，并在对应的脚本中将`CHECKPOINT_PATH`设置为文件夹路径。

## 结论

### [SuperGLUE](https://super.gluebenchmark.com)

dev set, 单模型, 单任务调优

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

## 入门

### Hugging Face Hub

你可以通过HuggingFace Hub访问GLM 模型。请安装`transformers>=4.23.1`，可以在[here](https://huggingface.co/models?filter=glm,thudm)找到所有可用模型.

#### 文本生成
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-10b", trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained("THUDM/glm-10b", trust_remote_code=True)
model = model.half().cuda()
model.eval()

# Inference推理
inputs = tokenizer("Ng is an adjunct professor at [MASK] (formerly associate professor and Director of its Stanford AI Lab or SAIL ). Also a pioneer in online education, Ng co-founded Coursera and deeplearning.ai.", return_tensors="pt")
inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
inputs = inputs.to('cuda')
outputs = model.generate(**inputs, max_length=512, eos_token_id=tokenizer.eop_token_id)
print(tokenizer.decode(outputs[0].tolist()))

# Training训练
inputs = tokenizer(
    ["Tsinghua University is located in [MASK].", "One minus one equals zero, is it correct? Answer: [MASK]"],
    return_tensors="pt", padding=True)
inputs = tokenizer.build_inputs_for_generation(inputs, targets=["Beijing", "No"], max_gen_length=8, padding=False)
inputs = inputs.to('cuda')
outputs = model(**inputs)
loss = outputs.loss
logits = outputs.logits
```
#### 文本分类
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
你也可以使用脚本`scripts/convert_glm_checkpoint_to_transformers.py`转换微调后的checkpoints. 
### Docker Image

我们准备了两个`docker`镜像，分别基于`CUDA 10.2`和`CUDA 11.2`构建。你可以从`Docker Hub`拉取预构建的镜像，并在`docker v19.03+`运行：

  ```shell
  docker run --gpus all --rm -it --ipc=host zxdu20/glm-cuda102
  ```

或者使用`glm-cuda112`替换`glm-cuda102`。

你也可以在[docker/cuda102.dockerfile](docker/cuda102.dockerfile)根据自己的需求修改镜像，然后构建你自己的镜像：

  ```shell
    docker build -f cuda102.dockerfile . -t glm-cuda102
  ```

### 手动安装

首先安装PyTorch (we use 1.7.0)和[apex](https://github.com/NVIDIA/apex), 然后通过 `pip install -r requirements.txt`命令安装其它依赖包。

### 克隆本仓库

  ```shell
  git clone https://github.com/THUDM/GLM
  cd GLM
  ```

### 模型并行

如果你遇到`CUDA out of memory`错误, 说明你的GPU 显存有限,你可以尝试模型并行，将参数分到多张GPU上。以模型双向并行为例。 首先运行 `change_mp.py` 对checkpoint进行拆分:

```shell
python change_mp.py path_to_the_checkpoint 2
```

然后在模型配置文件(比如[config_tasks/model_blocklm_10B.sh](config_tasks/model_blocklm_10B.sh))中更新checkpoint路径，在脚本(比如[scripts/ds_finetune_superglue.sh](scripts/ds_finetune_superglue.sh))中将`MP_SIZE`更改为`2`。

## 用法

在一些下游任务中，我们提供了微调GLM的脚本。

### 文本生成（从左到右） / 填空 (交互)

* 将`CHECKPOINT_PATH`更改为你的本地路径. 运行以下脚本

```
bash scripts/generate_block.sh \
     config_tasks/model_blocklm_10B_chinese.sh
```

一些模型（GLM-2B、GLM-10B和GLM-10B-Chinese）使用三种不同的掩码标记：`[mask]`用于短空白填充，`[sMASK]`用于句子填充，`[gMASK]`用来从左到右生成。


<details>
<summary><b>示例</b></summary>

#### `[MASK]`用法 (实体预测):

##### 示例1 (英文)

Context: Ng is an adjunct professor at [MASK] (formerly associate professor and Director of its Stanford AI Lab or SAIL ). Also a pioneer in online education, Ng co-founded Coursera and deeplearning.ai.

GLM: the stanford university

##### 示例2 (中文)

Context: 凯旋门位于意大利米兰市古城堡旁。1807年为纪念[MASK]而建，门高25米，顶上矗立两武士青铜古兵车铸像。

GLM:拿破仑军队攻克米兰城

#### `[sMASK]`用法 (句子预测)

##### 示例3 (英文)

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

##### 示例4 (中文)

Context: 工业互联网（Industrial
Internet）是新一代信息通信技术与工业经济深度融合的新型基础设施、应用模式和工业生态，通过对人、机、物、系统等的全面连接，构建起覆盖全产业链、全价值链的全新制造和服务体系，为工业乃至产业数字化、网络化、智能化发展提供了实现途径，是第四次工业革命的重要基石。[sMASK]
它以网络为基础、平台为中枢、数据为要素、安全为保障，既是工业数字化、网络化、智能化转型的基础设施，也是互联网、大数据、人工智能与实体经济深度融合的应用模式，同时也是一种新业态、新产业，将重塑企业形态、供应链和产业链。当前，工业互联网融合应用向国民经济重点行业广泛拓展，形成平台化设计、智能化制造、网络化协同、个性化定制、服务化延伸、数字化管理六大新模式，赋能、赋智、赋值作用不断显现，有力的促进了实体经济提质、增效、降本、绿色、安全发展。

GLM: 工业互联网是制造业技术、管理、模式的重大变革,是推动互联网、大数据、人工智能和实体经济深度融合的重要载体,是建设制造强国和网络强国的重要基础。

#### `[gMASK]`用法 (长文本生成)

##### 示例5 (中文)

Context: 问题：冬天，中国哪座城市最适合避寒？问题描述：能推荐一些国内适合冬天避寒的城市吗？回答用户：旅游爱好者 回答： [gMASK]

GLM:
海南三亚,三亚的冬天是三亚的旺季,因为冬天国内大部分地方都有雾霾,而三亚却是没有雾霾的地方,所以三亚是冬天最惬意不过的地方了。在东北长大的我觉得三亚简直就是一个天堂,冬天去海口三亚旅游,享受冬天,享受阳光沙滩。但是海口却是很干燥,需要多喝水。
三亚冬天最热门的玩法就是晒太阳,在海边晒晒太阳,在沙滩上晒晒太阳,感觉整个人都得到了解放。三亚还有一个特色项目,就是海上冲浪,在三亚的沙滩上冲浪也是一件非常刺激的事情。
海口,海口冬季的阳光十分温暖,海南的冬季也是属于冬季旅游的旺季。冬季的海口最棒的是去海南的热带野生动植物园,那里有数之不尽的热带小动物,在这里可以近距离的和它们接触,海南的热带野生动植物园也是海南的天然氧吧。还可以在海口观澜湖公园里感受海口美丽的海景。
贵阳,贵州的冬天也是十分温暖的,贵阳也是冬季避寒很好的城市之一。冬季去贵阳玩一定要去黔灵山,黔灵山是贵州香火很旺盛的一个寺庙,寺庙的冬季香火鼎盛,在冬季去寺庙游玩也是一个很好的体验。除了黔灵山,贵阳在冬季还有花溪公园可以去玩,花溪公园也是去当地公园玩最好的选择。
青岛,青岛的冬天是青岛最舒服的时候,青岛有很多海滨浴场,冬天去海边泡一泡温泉,然后晒晒太阳是一件十分惬意的事情。青岛也有沙滩,冬天在沙滩上晒晒太阳,看看海,再玩玩沙滩游戏,感觉十分快乐的事。
</details>

你也可以在单个示例中添加多个 `[MASK]` 和 `[sMASK]`. 模型将会从左至右依次填空. 每一个空白的答案都是从一个特定的标识符`<|startofpiece|>`开始。

<details>
<summary><b>示例</b></summary>

##### 示例1 (英文)

Context: There have been various types of pretraining architectures including autoencoding models (e.g., BERT), autoregressive models (e.g., GPT), and [MASK] (e.g., T5). [sMASK] We propose a General Language Model ( GLM) based on autoregressive blank infilling to address this challenge. GLM improves blank filling pretraining by adding 2D positional encodings and allowing an arbitrary order to predict spans, which results in performance gains over [MASK] on NLU tasks. Meanwhile, GLM can be pretrained for different types of tasks by varying the number and lengths of blanks. On a wide range of tasks across NLU, conditional and [MASK], GLM outperforms BERT, T5, and GPT given the same model sizes and data, and achieves the best performance from a single pretrained model with 1.25× parameters of BERT Large , demonstrating its generalizability to different downstream tasks.

GLM: <|startofpiece|> blank filling models<|startofpiece|> However, most of them cannot easily transfer to other downstream tasks due to the different characteristics of these tasks.<|startofpiece|> other pretrained models<|startofpiece|> unconditional reading, and semantic role labeling tasks

##### 示例2 (中文)

Context: 工业互联网（Industrial Internet）是新一代[MASK]与[MASK]深度融合的新型基础设施、应用模式和工业生态，通过对人、机、物、系统等的全面连接，构建起覆盖全产业链、全价值链的全新制造和服务体系，为工业乃至产业数字化、网络化、智能化发展提供了实现途径，是第四次工业革命的重要基石。[sMASK] 它以网络为基础、平台为中枢、数据为要素、安全为保障，既是工业数字化、网络化、智能化转型的基础设施，也是互联网、大数据、人工智能与实体经济深度融合的应用模式，同时也是一种新业态、新产业，将重塑企业形态、供应链和产业链。当前，工业互联网融合应用向国民经济重点行业广泛拓展，形成[MASK]、智能化制造、[MASK]、个性化定制、服务化延伸、数字化管理六大新模式，赋能、赋智、赋值作用不断显现，有力的促进了实体经济提质、增效、降本、绿色、安全发展。

GLM:
<|startofpiece|>信息技术(ICT)<|startofpiece|>工业经济(II2O)<|startofpiece|>我国工业互联网是面向工业全领域、全流程、全体系的互联网,具有多产业、多领域融合的特点。<|startofpiece|>网络化协同<|startofpiece|>平台企业

</details>


### SuperGLUE

- 下载[SuperGlue](https://super.gluebenchmark.com/tasks)数据并在
  [scripts/ds_finetune_superglue.sh](scripts/ds_finetune_superglue.sh)中检查实验设置. 注意，
  需要将 `DATA_ROOT, CHECKPOINT_PATH, SAVE_PATH`
  改为你的本地路径。根据你的硬件配置，你可能也需要更改`batch-size`和`nproc_per_node`。

- 运行以下脚本(use the COPA dataset as an example)

```
bash scripts/ds_finetune_superglue.sh \
     config_tasks/model_blocklm_10B.sh \
     config_tasks/task_copa.sh
```

- 我们还在代码中实现了[P-Tuning](https://arxiv.org/abs/2103.10385)。 运行下面的脚本可以集成p-tuning:

```shell
bash scripts/ds_finetune_superglue_prompt.sh \
     config_tasks/model_blocklm_10B.sh \
     config_tasks/task_copa.sh
```

- 要将GLM应用到一个具有完形填空微调方法、新的NLU数据集上, 在
  [tasks/superglue/dataset.py](tasks/superglue/dataset.py)中，实现了一个`DataProcessor`用于数据加载，在
  [tasks/superglue/pvp.py](tasks/superglue/pvp.py)中为完形填空问题添加`PVP`。更多详情请点击
  [这里](tasks/superglue/README.md).

### Seq2Seq

- 下载[Gigaword](https://github.com/harvardnlp/sent-summary)
  , [CNN/Daily Mail](https://github.com/artmatsak/cnn-dailymail)
  或[XSum](https://github.com/EdinburghNLP/XSum/tree/master/XSum-Dataset) 数据集，并在
  [scripts/ds_finetune_seq2seq.sh](scripts/ds_finetune_seq2seq.sh)中检查实验配置。 将`DATA_ROOT, CHECKPOINT_PATH, SAVE_PATH`改为你的本地路径。

- 运行下面的脚本 (使用`CNN/Daily Mail` 数据集作为样例)

  ```
  bash scripts/ds_finetune_seq2seq.sh \ 
     config_tasks/model_blocklm_10B.sh \ 
     config_tasks/seq_cnndm_org.sh
  ```
- 摘要信息写在 `./runs/experiment_name/test.jsonl.hyps`文件中。引用信息写在同一目录下的 `test.jsonl.refs` 文件里。 为了计算ROUGE,
  安装 [file2rouge](https://github.com/pltrdy/files2rouge) ，从[这里](http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip) 下载标准版 CoreNLP。运行以下脚本
  ```
  bash scripts/evaluate_seq2seq.sh \
   ./runs/experiment_name/test.jsonl.hyps ./runs/experiment_name/test.jsonl.refs
  ```

#### 使用自己的数据集进行训练

将你的seq2seq数据处理成`{split}.source` 和 `{split}.target`, 每一行都是样本的内容或标签, `split`可以是`train`, `val`, 以及`test`.

运行以下脚本：

```shell
bash scripts/ds_finetune_seq2seq.sh \ 
   config_tasks/model_blocklm_10B.sh \ 
   config_tasks/seq_customization.sh
```

你可以在`config_tasks/seq_customization.sh`
和 `config_tasks/config_blocklm_10B_cnndm.json`指定超参数。

### Multiple Choice (Zero-shot)

```shell
bash scripts/evaluate_multichoice.sh config_tasks/model_blocklm_10B.sh
```

注意， 需要将`CHECKPOINT_PATH` 和 `DATA_PATH` 改为你的本地路径。

数据文件每行的格式应为

```
{"inputs_pretokenized": "Context and question here", "choices_pretokenized": ["Choice 1", "Choice 2", "Choice 3"], "label": int}
```

### Language Modeling

#### LAMBADA Cloze Accuracy

* 下载[LAMBADA](https://github.com/cybertronai/bflm/blob/master/lambada_test.jsonl) 数据， 并在 [scripts/evaluate_lm.sh](scripts/evaluate_lm.sh) 修改
  `DATA_ROOT, CHECKPOINT_PATH` 
* 运行下面的脚本

```shell
bash scripts/evaluate_lm.sh \ 
     config_tasks/model_blocklm_large_generation.sh \
     config_tasks/zero_lambada.sh 
```

#### LM 困惑度

* 下载我们的 [test set of wikibook](https://mailstsinghuaeducn-my.sharepoint.com/:t:/g/personal/duzx16_mails_tsinghua_edu_cn/EQa_B6KY_q1FjtUeG-T52iMBFtNrfhfHcZbzMxfkJKXKRQ?e=inTdHh)
  或 [Wikitext103](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip) 数据集，并在 [scripts/evaluate_lm.sh](scripts/evaluate_lm.sh) 修改 `DATA_ROOT, CHECKPOINT_PATH`
* 运行下面的脚本
  ```shell
  bash scripts/evaluate_lm.sh \ 
     config_tasks/model_blocklm_large_generation.sh \
     config_tasks/zero_wikitext.sh 
  ```

### 文本填充

- 下载[Yahoo](https://github.com/Varal7/blank_language_model) 数据集，并在[scripts/finetune_blank.sh](scripts/finetune_blank.sh)中检查实验配置。 将 `DATA_ROOT, CHECKPOINT_PATH, SAVE_PATH` 修改为你的本地路径。

- 运行以下脚本

```
bash scripts/finetune_blank.sh \ 
     config_tasks/model_blocklm_large.sh \ 
     config_tasks/seq_blank.sh
```

## 预训练Pretrain

运行以下脚本来预训练GLM大模型：

```shell
bash scripts/ds_pretrain_nvidia.sh config/ds_block_large.sh
```

脚本 [scripts/ds_pretrain_nvidia.sh](scripts/ds_pretrain_nvidia.sh) 使用 DeepSpeed启动训练程序。
你可以通过 `NUM_WORKERS` 和 `NUM_GPUS_PER_WORKER` 参数修改workers数量以及每个 worker的gpu数量.
也可以通过修改 `HOST_FILE_PATH` 参数来更改OpenMPI-style hostfile的路径。关于DeepSpeed启动器的更多详情请参阅 [这里](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node).

[config/ds_block_large.sh](config/ds_block_large.sh) 文件定义了预训练的超参数。 大部分的参数是不言自明的。 特别的, 
在 [data_utils/corpora.py](data_utils/corpora.py)中的`NAMED_CORPORA`参数里可以为`--train-data`定义多个关键字。 优化器的超参数定义在`config`文件夹下相对应的json文件里。 json文件中的参数含义可以在 [这里](https://www.deepspeed.ai/docs/config-json)找到。

## 引用

部分代码是基于 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
和 [PET](https://github.com/timoschick/pet).

如果我们的代码对您有用，请引用我们的论文:

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
