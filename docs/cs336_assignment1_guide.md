# CS336 Assignment 1 学生指引：有限条件下完成 Basics

资料来源：本地 `assignment1-basics/cs336_assignment1_basics.pdf`，官方仓库 <https://github.com/stanford-cs336/assignment1-basics>，课程页 <https://cs336.stanford.edu/>。注意：Stanford 官方作业说明允许 AI 用于高层概念解释，但不允许在正式选课提交中用 AI 完成实现；如果你是正式学生，请把本文当作路线图和自查清单，而不是直接提交材料。

## 总体背景

CS336 Assignment 1 的目标是从零搭出一个可训练的 decoder-only Transformer 语言模型。你需要先实现 byte-level BPE tokenizer，把原始文本转成 token id；再实现 Transformer LM 的核心层，包括 Linear、Embedding、RMSNorm、SwiGLU、RoPE、causal multi-head self-attention、Transformer block 和完整 LM；最后实现训练相关组件，包括 softmax、cross entropy、AdamW、学习率调度、梯度裁剪、batch 采样、checkpoint、训练循环、解码和实验日志。

仓库结构很简单：`cs336_basics/` 是你写代码的地方，`tests/adapters.py` 是测试和你的实现之间的胶水，`tests/test_*.py` 是验收标准。作业要求尽量从零实现，不能用 `torch.nn.Linear`、`torch.nn.Embedding`、`torch.nn.functional.softmax`、`torch.optim.AdamW` 这类现成实现；可以用 `torch.nn.Parameter`、`torch.nn.Module`、`torch.optim.Optimizer` 基类、普通 PyTorch 张量操作、`einops`、`regex`。

有限算力策略是先让单元测试全绿，再只跑缩小版训练。CPU/MPS 用户可以把 TinyStories 训练总 token 数从官方的大约 327M 降到约 40M，例如 `batch_size * steps * context_length = 32 * 5000 * 256`。

## 0. 环境和工作方式

### 问题背景

仓库使用 `uv` 管理环境。初始状态下，`tests/adapters.py` 中所有函数都会抛 `NotImplementedError`，所以测试失败是正常的。

### 学生要做的事情

1. 在 `cs336_basics/` 下新建自己的模块，例如 `model.py`、`tokenizer.py`、`optimizer.py`、`training.py`。
2. 在 `tests/adapters.py` 里只写胶水代码，调用你的模块，不要把大量逻辑塞进 adapters。
3. 分阶段跑测试：

```bash
uv run pytest tests/test_model.py -k "linear or embedding or rmsnorm"
uv run pytest tests/test_model.py
uv run pytest tests/test_nn_utils.py tests/test_optimizer.py tests/test_data.py tests/test_serialization.py
uv run pytest tests/test_train_bpe.py tests/test_tokenizer.py
uv run pytest
```

### 相关答案

最终你至少应有这些接口：

```text
cs336_basics/
  model.py          # Linear, Embedding, RMSNorm, SwiGLU, RoPE, Attention, Block, LM
  tokenizer.py      # train_bpe, Tokenizer
  optimizer.py      # AdamW, lr schedule, gradient clipping
  training.py       # get_batch, save/load checkpoint, train loop, decode
tests/adapters.py   # 调用上面实现
```

## 1. Unicode 与 UTF-8

### 问题背景

BPE tokenizer 不是直接在 Unicode code point 上训练，而是先把字符串编码为 UTF-8 bytes。这样初始词表固定为 256 个 byte，不会有 out-of-vocabulary 问题。

### 学生要做的事情

回答 `unicode1` 和 `unicode2` 的概念题。建议在 Python REPL 里亲自运行：

```python
chr(0)
repr(chr(0))
print("a" + chr(0) + "b")
"こんにちは".encode("utf-8")
"こんにちは".encode("utf-16")
"こんにちは".encode("utf-32")
```

### 相关答案

`chr(0)` 是 null character，`repr` 通常显示为 `'\x00'`，直接打印时不可见但占一个字符位置。UTF-8 相比 UTF-16/UTF-32 更适合作为 byte-level tokenizer 的底层编码，因为 ASCII 和英文文本更紧凑，并且互联网文本主要使用 UTF-8。逐 byte 解码 UTF-8 是错的，因为多字节字符的单个 byte 不是完整字符，例如 `"é".encode("utf-8") == b"\xc3\xa9"`，单独解码 `b"\xc3"` 会失败。无效两字节序列示例：`b"\x80\x80"` 或 `b"\xc3\x28"`。

## 2. BPE Tokenizer Training

### 问题背景

BPE 的训练过程是：初始化 256 个 byte token 加 special tokens；使用 GPT-2 风格正则预分词；把每个 pre-token 变成 byte tuple；反复统计相邻 token pair，选最高频 pair，频率相同取字典序更大的 pair；合并它并加入词表，直到达到 `vocab_size`。

特殊 token 在训练时是硬边界：它们要加入词表，但不参与 pair 统计，也不能和普通文本合并。

### 学生要做的事情

先实现一个简单版本，通过小测试；再优化 pre-tokenization 和 pair 统计。测试重点：

```bash
uv run pytest tests/test_train_bpe.py
```

关键约束：

```python
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

训练时要用 `regex` 包，不是 Python 内置 `re`。同频 tie-break 用：

```python
best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
```

### 相关答案

一个可接受的实现框架：

```python
def train_bpe(input_path, vocab_size, special_tokens):
    vocab = {i: bytes([i]) for i in range(256)}
    for s in special_tokens:
        if s.encode("utf-8") not in vocab.values():
            vocab[len(vocab)] = s.encode("utf-8")

    word_counts = pretokenize_file(input_path, special_tokens)
    merges = []
    while len(vocab) < vocab_size:
        pair_counts = count_pairs(word_counts)
        if not pair_counts:
            break
        pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(pair)
        vocab[len(vocab)] = pair[0] + pair[1]
        word_counts = merge_pair_in_counts(word_counts, pair)
    return vocab, merges
```

有限条件建议：先在 `tests/fixtures/corpus.en` 上调通，再在 TinyStories validation 上跑，最后再上完整 TinyStories。OpenWebText 32K BPE 对 CPU/RAM 压力更大，重点是给出可复现实验日志。

## 3. BPE Tokenizer Encode/Decode

### 问题背景

训练 BPE 得到的是 `vocab` 和 `merges`。Tokenizer 要把文本编码成 token id，并能把 id 解码回文本。特殊 token 在编码时必须保持为单个 token，尤其要处理重叠特殊 token，例如 `<|endoftext|>` 和 `<|endoftext|><|endoftext|>`，应优先匹配更长的。

### 学生要做的事情

实现：

```python
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None): ...
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None): ...
    def encode(self, text: str) -> list[int]: ...
    def encode_iterable(self, iterable): ...
    def decode(self, ids: list[int]) -> str: ...
```

验收：

```bash
uv run pytest tests/test_tokenizer.py
```

### 相关答案

编码每个 pre-token 时，不要每次从头扫描所有 merges 做低效替换；最简单稳妥的做法是建立 `rank = {merge_pair: index}`，在当前 token 序列里找 rank 最小的可合并 pair，合并后继续，直到没有 pair 可合并。

`decode` 用：

```python
return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
```

`encode_iterable` 对大文件要分块或逐段处理；最安全的低内存实现是逐行编码并 yield。但如果文件没有换行，逐行仍可能吃内存，进阶版要按 special token 或 chunk 边界处理。

## 4. Transformer 基础层

### 问题背景

模型是标准 pre-norm decoder-only Transformer：token embedding 后经过多层 block，每层是 causal self-attention with RoPE 和 SwiGLU FFN，最后 RMSNorm 和 LM head 输出 logits。

### 学生要做的事情

按依赖顺序实现：

1. `Linear`: 权重形状 `(out_features, in_features)`，前向 `x @ W.T`。
2. `Embedding`: 权重形状 `(vocab_size, d_model)`，前向索引。
3. `RMSNorm`: 对最后一维做 RMS normalization，计算时 upcast 到 float32。
4. `SiLU`: `x * sigmoid(x)`。
5. `SwiGLU`: `W2(SiLU(W1 x) * W3 x)`。

验收：

```bash
uv run pytest tests/test_model.py -k "linear or embedding or rmsnorm or silu or swiglu"
```

### 相关答案

核心公式：

```python
linear = x @ weight.T
embedding = weight[token_ids]
rmsnorm = x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * gain
silu = x * torch.sigmoid(x)
swiglu = (silu(x @ w1.T) * (x @ w3.T)) @ w2.T
```

## 5. Attention、RoPE、Block、LM

### 问题背景

Scaled dot-product attention：

```text
Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k) + mask) V
```

decoder-only LM 必须使用 causal mask，保证位置 `t` 不能看未来 token。RoPE 只作用于 Q 和 K，不作用于 V。

### 学生要做的事情

实现：

1. `scaled_dot_product_attention(Q, K, V, mask=None)`
2. `RoPE`
3. `MultiHeadSelfAttention`
4. `TransformerBlock`
5. `TransformerLM`

验收：

```bash
uv run pytest tests/test_model.py -k "attention or rope or transformer"
```

### 相关答案

形状约定：

```text
x:       (..., seq, d_model)
q/k/v:   (..., heads, seq, head_dim)
attn:    (..., heads, seq, seq)
out:     (..., seq, d_model)
```

RoPE 常见实现：

```python
freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2) / d_k))
angles = token_positions[..., None] * freqs
x_even, x_odd = x[..., 0::2], x[..., 1::2]
y_even = x_even * cos - x_odd * sin
y_odd = x_even * sin + x_odd * cos
```

Transformer block 是 pre-norm：

```python
z = x + attn(ln1(x))
y = z + ffn(ln2(z))
```

完整 LM：

```python
x = token_embeddings(input_ids)
for layer in layers:
    x = layer(x)
x = ln_final(x)
logits = lm_head(x)
```

## 6. Loss、AdamW、LR Schedule、Gradient Clipping

### 问题背景

训练组件必须数值稳定。softmax 和 cross entropy 都要减去最大 logit；AdamW 要实现 bias correction 和 decoupled weight decay；梯度裁剪按所有参数梯度的全局 L2 norm 缩放。

### 学生要做的事情

实现并测试：

```bash
uv run pytest tests/test_nn_utils.py tests/test_optimizer.py
```

### 相关答案

```python
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
cross_entropy = mean(logsumexp(logits) - logits[range(batch), targets])
lr(it) = warmup 线性上升；之后 cosine decay；超过 cycle 后固定 min_lr
clip_coef = max_l2_norm / (total_norm + 1e-6)
```

AdamW 单步：

```python
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad * grad
lr_t = lr * sqrt(1 - beta2**t) / (1 - beta1**t)
param -= lr_t * m / (sqrt(v) + eps)
param -= lr * weight_decay * param
```

## 7. Data Loading 与 Checkpointing

### 问题背景

语言模型训练 batch 是从一维 token id 数组中随机取连续片段，输入 `x` 是长度 `context_length`，标签 `y` 是向右偏移一位。

### 学生要做的事情

实现：

```python
def get_batch(dataset, batch_size, context_length, device):
    # x[i] = dataset[start:start+context_length]
    # y[i] = dataset[start+1:start+context_length+1]
```

checkpoint 要保存 model state、optimizer state 和 iteration。

### 相关答案

```python
starts = torch.randint(0, len(dataset) - context_length, (batch_size,))
x = torch.stack([torch.as_tensor(dataset[i:i+context_length]) for i in starts])
y = torch.stack([torch.as_tensor(dataset[i+1:i+context_length+1]) for i in starts])
```

checkpoint：

```python
torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}, out)
state = torch.load(src, map_location="cpu")
model.load_state_dict(state["model"])
optimizer.load_state_dict(state["optimizer"])
return state["iteration"]
```

## 8. 训练循环与有限算力实验

### 问题背景

训练循环把所有组件串起来：采样 batch、forward、loss、backward、clip、optimizer step、lr schedule、验证、checkpoint、日志。作业还要求做 TinyStories 和 OWT 实验、学习率/批大小 sweep、生成文本、架构消融。

### 学生要做的事情

最小训练循环：

```python
for it in range(max_iters):
    lr = get_lr(it, ...)
    for group in optimizer.param_groups:
        group["lr"] = lr
    x, y = get_batch(train_ids, batch_size, context_length, device)
    logits = model(x)
    loss = cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    clip_gradients(model.parameters(), max_norm)
    optimizer.step()
```

有限算力建议：

```text
debug:      batch=4,  context=64,  steps=100
CPU/MPS:    batch=32, context=256, steps=5000
full-ish:   batch=64, context=256, steps=20000 或按预算调整
```

### 相关答案

必须在 writeup 里记录：

```text
机器/设备、数据、vocab_size、模型配置、batch/context/steps、学习率、warmup、AdamW 参数、
训练/验证 loss 曲线、训练耗时、吞吐 tokens/sec、生成样本、消融实验结论。
```

## 9. 实验问题怎么答

### 问题背景

作业里的很多问题不是固定数值题，而是要求你实际运行并报告曲线或结果。因此答案应以你自己的代码、机器、随机种子和数据版本为准。

### 学生要做的事情

至少保存：

```text
runs/
  ts_base/config.json
  ts_base/metrics.csv
  ts_base/checkpoint.pt
  ts_base/sample.txt
  ts_lr_sweep/*.csv
  ablations/*.csv
```

### 相关答案

可以使用这种报告模板：

```markdown
配置：vocab=10000, d_model=512, layers=4, heads=16, d_ff=1344, context=256,
batch=32, steps=5000, AdamW betas=(0.9,0.95), wd=0.1, max_lr=3e-4,
warmup=500, min_lr=3e-5, device=mps。

结果：训练 loss 从 X 降到 Y，验证 loss 为 Z，吞吐约 T tokens/sec。
学习率过大时在第 N 步发散；最佳学习率接近但低于发散边界。
```

不要把网上或别人机器的数值当成你的提交结果。实验题的“答案”核心是曲线、日志和合理解释。

