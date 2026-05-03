# CS336 Spring 2025 作业1：基础

这是一个中文学习资料仓库。

完整作业说明请参阅 [cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)。

如果你发现作业说明或代码中有任何问题，请随时提交 GitHub issue 或 pull request 来修复。

## 环境配置

### 环境管理

我们使用 `uv` 来管理环境，以确保可重现性、可移植性和易用性。
请在此处安装 `uv` [安装指引](https://github.com/astral-sh/uv#installation)（推荐），或者运行 `pip install uv` / `brew install uv`。
建议你阅读一下 `uv` 项目管理的相关文档 [链接](https://docs.astral.sh/uv/guides/projects/#managing-dependencies)（你不会后悔的！）。

现在你可以使用以下命令运行仓库中的任何代码：
```sh
uv run <python文件路径>
```
环境会被自动解析并在需要时激活。

### 运行单元测试

```sh
uv run pytest
```

初始状态下，所有测试应该都会因为 `NotImplementedError` 而失败。
要连接你的实现到测试，请完成 [./tests/adapters.py](./tests/adapters.py) 中的函数。

### 下载数据

下载 TinyStories 数据和 OpenWebText 的子样本：

```sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```