# CS336 Assignment 1 Learning

这是一个面向 CS336 Assignment 1 的中文学习与实现仓库，目标是把作业一涉及的 tokenizer、decoder-only Transformer、loss、optimizer、checkpoint 和训练实验串成一条完整学习路径。

本仓库保留源码、学习文档、基础项目配置和轻量测试代码；不上传本地数据集、训练 checkpoint、token cache、模型产物和官方大型测试 fixture。

## 内容概览

| 路径 | 内容 |
|------|------|
| `cs336_basics/` | 作业一核心实现：模型、tokenizer、训练工具 |
| `docs/cs336_assignment1_learning_guide.md` | 大型中文学习文档：从文本到可训练 GPT |
| `docs/cs336_assignment1_guide.md` | 作业一完成路线与自查清单 |
| `docs/optimizer_learning_guide.md` | optimizer、loss、checkpoint 学习文档 |
| `docs/assets/` | 学习文档配图 |
| `tests/` | 作业测试适配层与轻量测试代码 |
| `train.py` | 本地训练脚本 |
| `generate_story.py` | 文本生成脚本 |
| `*_implementation_guide.md` | 分模块实现笔记 |
| `cs336_assignment1_basics.pdf` | 作业一说明 PDF |

## 推荐阅读顺序

1. 先读 [docs/cs336_assignment1_learning_guide.md](docs/cs336_assignment1_learning_guide.md)，建立完整知识地图。
2. 再看 `cs336_basics/tokenizer.py`，理解 byte-level BPE 的训练、编码和解码。
3. 接着看 `cs336_basics/model.py`，把 Linear、Embedding、RMSNorm、SwiGLU、Attention、RoPE 和 TransformerLM 串起来。
4. 最后看 `cs336_basics/training.py`、`train.py` 和 `generate_story.py`，理解 loss、AdamW、学习率、checkpoint、训练循环与采样生成。

## 环境配置

本项目使用 `uv` 管理 Python 环境：

```bash
uv sync
```

运行测试：

```bash
uv run pytest
```

说明：为了让仓库保持轻量，本仓库不包含官方大型 snapshots、TinyStories 5M fixture 和 TorchScript 模型 fixture。如果需要完整复现官方测试，请从 Stanford CS336 Assignment 1 官方仓库获取完整测试资源。

## 数据与训练产物

以下内容不会提交到仓库：

- `data/` 下的 TinyStories / OpenWebText 数据
- `out/`、`runs/`、`checkpoints/` 下的训练输出
- `.venv/`、`.pytest_cache/`、`__pycache__/` 等本地缓存
- `*.pt`、`*.bin`、`*.pkl`、`*.npz` 等模型、缓存和快照文件

如需本地训练，可自行下载数据：

```bash
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz
```

## 学习主线

```text
文本
-> UTF-8 bytes
-> BPE token ids
-> x/y batch
-> decoder-only Transformer
-> logits
-> cross entropy loss
-> gradients
-> AdamW update
-> checkpoint
-> experiment report
```

这条主线对应作业一的核心能力：从原始文本出发，搭建一个可以训练和生成的小型 GPT 风格语言模型。

## 参考资料

- [Stanford CS336](https://cs336.stanford.edu/)
- [Assignment 1 官方仓库](https://github.com/stanford-cs336/assignment1-basics)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
