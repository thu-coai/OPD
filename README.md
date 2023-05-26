
![图片](https://lingyou-1302942961.cos.ap-beijing.myqcloud.com/lingyou/166747137310657482761-5415-450b-a792-701f66b87229.png)
------




<p align="center">
  <a href="">Demo(微信公众号)</a> •
  <a href="">博客</a> •
</p>


# OPD：中文开放域对话预训练模型

OPD是一个中文开放域对话预训练模型，拥有63亿参数，在70GB高质量对话数据上进行训练而成。它具有如下优势：

- **大规模**：OPD的模型参数量为6.3B，是目前世界上规模最大的开源中文对话预训练模型

- **高性能**：我们通过自动评测和人工评测来全面评估OPD的性能。评测结果显示，OPD兼顾出色的闲聊能力与知识问答能力。得益于此，OPD的**多轮交互能力突出**，能够与用户进行多轮、深入的对话交互，性能显著优于EVA2.0, PLATO和PANGU-BOT，更受用户偏爱。

- **开源开放**：我们后续计划逐步开源**一系列中文对话模型相关生态**，推动中文对话领域的发展。具体包括：

  - **世界上最大的开源中文对话预训练模型**：OPD

  - **多维度中文对话评价模型**：[对话信息量](https://huggingface.co/thu-coai/roberta-zh-specific)、[相关性](https://huggingface.co/thu-coai/roberta-zh-sensible)、[一致性](https://huggingface.co/thu-coai/roberta-base-cdconv)、[安全性](https://huggingface.co/thu-coai/roberta-base-cold?text=%E6%88%91%E5%96%9C%E6%AC%A2%E4%BD%A0%E3%80%82+%E6%88%91%E7%88%B1%E4%BD%A0)等多个维度各自的评价模型。

![Alt Text](pic/multiturn.gif)

![图片](https://lingyou-1302942961.cos.ap-beijing.myqcloud.com/lingyou/1667550036683b4d9d64c-b8d9-463d-b06b-35648a84f323.png)


## 参数下载

OPD模型可从[此处](https://cloud.tsinghua.edu.cn/d/ea490ba85640419785b5/)下载

下载完成后，需将拆分后的参数文件合并。

假设下载后的参数文件路径为`results/opd`, 可按如下方式合并
```
cd src
python tools/merge_checkpoint.py --ckpt_path ../results/opd
```

## 环境配置

- python 3.8, cuda 10.2

- `pip install -r requirements.txt`

## 运行代码

在运行前，需将脚本中的`PROJECT_DIR`, `CKPT_PATH`等路径根据实际情况进行修改

### 交互


```bash
cd src
bash scripts/interactive.sh
```

### 静态生成

- 数据格式: 每行一个session, 包含N个utterance, 用`\t`分隔。前N-1个utterance作为context输入
- 执行生成
```bash
cd bsah
bash scripts/inference_static.sh

# 关键参数:
# 输入路径: $TEST_FILE
# 输出路径: $OUT_FILE
# 模型文件: $CKPT
```


### 训练

#### 准备训练数据

1. 截断数据
```bash
bash scripts/prepare_data.sh

# 关键参数:
# INPUT_PATH: 输入数据路径。输入数据的格式为每行一个context-response pair, 用\t分隔
# OUTPUT_PATH: 输出的目录，输出文件会放置在${OUTPUT_PATH}/data.txt中。输出数据的格式为 每行一个dict, 包含source和target两个字段，分别代表context和response。
# max_seq_len: 截断长度，默认设置为512
```

2. tokenize
```bash
bash scripts/encode_data.sh

# 关键参数:
# INPUT_PATH: 输入数据，即上一步的输出文件
# OUTPUT_PATH: 输出的目录。执行完成后会新增四个文件, dialog_context_0.bin, dialog_context_0.idx, dialog_target_0.bin, dialog_target_0.idx
```

#### 开始训练

```bash
bash scripts/train.sh

# 关键参数:
# GPUS_PER_NODE: 单机的卡数
# DATASET: 数据路径。${DATASET}/train, ${DATASET}/valid两个文件夹分别存放了处理好的训练集和验证集
# --load: 是否load参数
```

## 引用

```
@misc{opd2023,
    title = {OPD: A Chinese Open-Domain Dialogue Pre-trained Model},
    url = {http://coai.cs.tsinghua.edu.cn/static/opd/posts/opd_blog/},
    author = {Jiaxin Wen and Yi Song and Pei Ke and Minlie Huang},
    month = {May},
    year = {2023}
}
```