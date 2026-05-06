# CS336 Assignment 1 Transformer Model 实现专题文档

本文只讲 Assignment 1 的 Transformer Model 部分：`Linear`、`Embedding`、`RMSNorm`、`SiLU`、`SwiGLU`、`scaled_dot_product_attention`、`RoPE`、`MultiHeadSelfAttention`、`TransformerBlock`、`TransformerLM`，以及它们如何接到 `tests/adapters.py` 并通过 `tests/test_model.py`。

本地当前状态备注：`cs336_basics/model.py` 里已经有 `Linear` 和 `Embedding` 的雏形，但目前使用了 `torch.nn.Linear` / `torch.nn.Embedding` 包装，后续最好改成作业要求的 `nn.Parameter` + 张量运算实现。`RMSNorm` 还没有完成，`tests/adapters.py` 中 Transformer 相关的大多数 adapter 仍然是 `NotImplementedError`。

## 1. Transformer Model 测试架构

### 问题背景

CS336 的模型测试不会直接关心你内部怎么拆文件。测试只认识 `tests/adapters.py` 里的这些入口：

```python
run_linear(...)
run_embedding(...)
run_swiglu(...)
run_scaled_dot_product_attention(...)
run_multihead_self_attention(...)
run_multihead_self_attention_with_rope(...)
run_rope(...)
run_transformer_block(...)
run_transformer_lm(...)
run_rmsnorm(...)
run_silu(...)
```

也就是说，测试文件准备好输入张量、参考权重和 snapshot，然后调用 adapter；adapter 再调用你在 `cs336_basics/model.py` 里写的真实实现。

调用链是：

```text
tests/test_model.py
  -> tests/adapters.py::run_linear
    -> cs336_basics/model.py::Linear

tests/test_model.py
  -> tests/adapters.py::run_multihead_self_attention_with_rope
    -> cs336_basics/model.py::MultiHeadSelfAttention
      -> RotaryPositionalEmbedding
      -> scaled_dot_product_attention

tests/test_model.py
  -> tests/adapters.py::run_transformer_lm
    -> cs336_basics/model.py::TransformerLM
      -> TransformerBlock
      -> MultiHeadSelfAttention + RMSNorm + SwiGLU
```

### 学生要做的事情

你不需要改测试逻辑。你只需要：

1. 在 `cs336_basics/model.py` 中实现模型组件。
2. 在 `tests/adapters.py` 中把每个 `run_*` 函数接到你的实现。
3. 分阶段跑 `tests/test_model.py`，先小模块，后完整 LM。

推荐测试顺序：

```bash
.venv\Scripts\python.exe -m pytest tests/test_model.py -k "linear or embedding or rmsnorm or silu or swiglu" -q
.venv\Scripts\python.exe -m pytest tests/test_model.py -k "attention or rope" -q
.venv\Scripts\python.exe -m pytest tests/test_model.py -k "transformer_block or transformer_lm" -q
.venv\Scripts\python.exe -m pytest tests/test_model.py -q
```

如果 `uv` 可用，也可以用：

```bash
uv run pytest tests/test_model.py -q
```

### 相关答案

adapter 里不应该写 substantive logic。它只负责实例化模块、塞入测试给定的权重、调用 forward。例如：

```python
def run_linear(d_in, d_out, weights, in_features):
    from cs336_basics.model import Linear
    layer = Linear(d_in, d_out, device=weights.device, dtype=weights.dtype)
    layer.weight.data.copy_(weights)
    return layer(in_features)
```

注意：如果你的 `Linear` 内部属性叫 `self.linear.weight`，adapter 就要写 `layer.linear.weight.data.copy_(weights)`。但更推荐按作业要求让 `Linear.weight` 直接是 `nn.Parameter`，这样 state dict key 会更干净，也更容易和后面的 `load_state_dict` 对齐。

## 2. 相关测试文件讲解

### 问题背景

Transformer Model 部分主要涉及这些文件：

```text
tests/test_model.py                  # 模型组件测试
tests/adapters.py                    # 测试入口胶水
tests/conftest.py                    # 输入 shape、随机张量、fixture 配置
tests/fixtures/ts_tests/model.pt     # 参考模型权重
tests/fixtures/ts_tests/model_config.json
tests/_snapshots/*.npz               # 参考输出 snapshot
cs336_basics/model.py                # 你实现模型的地方
```

`tests/test_model.py` 的判断方式大多是 snapshot testing：给你固定输入和固定权重，要求你的输出和 `tests/_snapshots/*.npz` 里的参考数组在容差内一致。

### 学生要做的事情

阅读测试时优先看每个 `run_*` 的参数和期望输出 shape。当前 fixture 的核心配置是：

```text
batch_size = 4
n_queries = 12
n_keys = 16
n_heads = 4
d_head = 16
d_model = 64
d_ff = 128
theta = 10000.0
vocab_size = 10000
n_layers = 3
```

这些 shape 会贯穿所有测试：

```text
in_embeddings: (batch, seq, d_model) = (4, 12, 64)
q/k/v:         (batch, seq, d_model)
token ids:     (batch, seq)
LM logits:     (batch, seq, vocab_size)
```

### 相关答案

`tests/test_model.py` 中的测试大致分成三层：

1. 基础层：

```text
test_linear
test_embedding
test_rmsnorm
test_silu_matches_pytorch
test_swiglu
```

这层只检查单个函数或小模块。先让这些通过，后面的问题会少很多。

2. Attention 和 RoPE：

```text
test_scaled_dot_product_attention
test_4d_scaled_dot_product_attention
test_rope
test_multihead_self_attention
test_multihead_self_attention_with_rope
```

这里最容易错的是维度顺序、mask 的 bool 语义、RoPE 的 broadcast。

3. 组合模块：

```text
test_transformer_block
test_transformer_lm
test_transformer_lm_truncated_input
```

这层要求你的模块命名、权重加载、pre-norm 顺序、causal mask、RoPE 位置都一致。

## 3. 文件和类规划

### 问题背景

Transformer Model 部分可以全部放在一个文件：

```text
cs336_basics/model.py
```

不要一开始就写完整 `TransformerLM`。正确顺序是先保证小模块能用，再把它们组合起来。

### 学生要做的事情

推荐函数和类顺序：

```python
class Linear(nn.Module): ...
class Embedding(nn.Module): ...
class RMSNorm(nn.Module): ...

def silu(x): ...

class SwiGLU(nn.Module): ...

def softmax(x, dim): ...
def scaled_dot_product_attention(Q, K, V, mask=None): ...

class RotaryPositionalEmbedding(nn.Module): ...
class MultiHeadSelfAttention(nn.Module): ...
class TransformerBlock(nn.Module): ...
class TransformerLM(nn.Module): ...
```

### 相关答案

依赖关系如下：

```text
Linear
  -> SwiGLU
  -> MultiHeadSelfAttention
  -> TransformerLM.lm_head

Embedding
  -> TransformerLM.token_embeddings

RMSNorm
  -> TransformerBlock.ln1 / ln2
  -> TransformerLM.ln_final

RotaryPositionalEmbedding
  -> MultiHeadSelfAttention with RoPE

scaled_dot_product_attention
  -> MultiHeadSelfAttention

TransformerBlock
  -> TransformerLM
```

建议把真正的数学逻辑写在 `model.py`，adapter 只做转发。这样后面写训练循环时也能直接复用同一套模型。

## 4. 第一步：Linear 和 Embedding

### 问题背景

作业希望你自己实现 Linear 和 Embedding，而不是直接用 `torch.nn.Linear` 或 `torch.nn.Embedding`。可以使用 `torch.nn.Module` 和 `torch.nn.Parameter`。

`Linear` 的权重 shape 是：

```text
weight: (d_out, d_in)
input:  (..., d_in)
output: (..., d_out)
```

前向公式：

```python
output = input @ weight.T
```

`Embedding` 的权重 shape 是：

```text
weight:    (vocab_size, d_model)
token_ids: (...)
output:    (..., d_model)
```

前向就是高级索引：

```python
output = weight[token_ids]
```

### 学生要做的事情

实现：

```python
class Linear(nn.Module):
    def __init__(self, d_in, d_out, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        # 初始化可以用 trunc_normal_

    def forward(self, in_features):
        return in_features @ self.weight.T
```

```python
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

    def forward(self, token_ids):
        return self.weight[token_ids]
```

### 相关答案

adapter 对应写法：

```python
def run_linear(d_in, d_out, weights, in_features):
    from cs336_basics.model import Linear
    layer = Linear(d_in, d_out, device=weights.device, dtype=weights.dtype)
    layer.weight.data.copy_(weights)
    return layer(in_features)


def run_embedding(vocab_size, d_model, weights, token_ids):
    from cs336_basics.model import Embedding
    layer = Embedding(vocab_size, d_model, device=weights.device, dtype=weights.dtype)
    layer.weight.data.copy_(weights)
    return layer(token_ids)
```

常见错误：

```python
# 错误：少了转置
return in_features @ self.weight

# 错误：用 one-hot 再矩阵乘法，慢而且容易 dtype 错
one_hot = torch.nn.functional.one_hot(token_ids, vocab_size)
return one_hot @ self.weight
```

## 5. 第二步：RMSNorm

### 问题背景

RMSNorm 只在最后一维做归一化，不减均值。公式是：

```text
y = x / sqrt(mean(x^2) + eps) * weight
```

其中 `weight` 是可训练参数，shape 是 `(d_model,)`。

### 学生要做的事情

实现时建议先把输入 upcast 到 `float32`，算完后再 cast 回输入 dtype。这对 fp16/bf16 训练更稳，也符合作业常见预期。

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        in_dtype = x.dtype
        x_f = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x_f * x_f, dim=-1, keepdim=True) + self.eps)
        return (x_f / rms * self.weight).to(in_dtype)
```

### 相关答案

adapter：

```python
def run_rmsnorm(d_model, eps, weights, in_features):
    from cs336_basics.model import RMSNorm
    layer = RMSNorm(d_model, eps=eps, device=weights.device, dtype=weights.dtype)
    layer.weight.data.copy_(weights)
    return layer(in_features)
```

常见错误：

```python
# 错误：这是 LayerNorm，不是 RMSNorm
x = x - x.mean(dim=-1, keepdim=True)

# 错误：mean 维度不对
torch.mean(x * x)

# 错误：weight shape 写成 (1, d_model) 可能能 broadcast，但 state_dict 不匹配
self.weight = nn.Parameter(torch.ones(1, d_model))
```

## 6. 第三步：SiLU 和 SwiGLU

### 问题背景

SiLU 公式：

```python
silu(x) = x * sigmoid(x)
```

SwiGLU 是带门控的 FFN：

```text
SwiGLU(x) = W2(SiLU(W1 x) * W3 x)
```

权重 shape：

```text
w1.weight: (d_ff, d_model)
w2.weight: (d_model, d_ff)
w3.weight: (d_ff, d_model)
```

### 学生要做的事情

实现：

```python
def silu(x):
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))
```

### 相关答案

adapter：

```python
def run_silu(in_features):
    from cs336_basics.model import silu
    return silu(in_features)


def run_swiglu(d_model, d_ff, w1_weight, w2_weight, w3_weight, in_features):
    from cs336_basics.model import SwiGLU
    layer = SwiGLU(d_model, d_ff, device=in_features.device, dtype=in_features.dtype)
    layer.w1.weight.data.copy_(w1_weight)
    layer.w2.weight.data.copy_(w2_weight)
    layer.w3.weight.data.copy_(w3_weight)
    return layer(in_features)
```

常见错误：

```python
# 错误：w2 的输入输出写反
self.w2 = Linear(d_model, d_ff)

# 错误：门控乘法位置不对
return self.w2(silu(self.w1(x))) * self.w3(x)
```

## 7. 第四步：Scaled Dot-Product Attention

### 问题背景

Attention 公式：

```text
scores = Q K^T / sqrt(d_k)
attn = softmax(scores)
out = attn V
```

测试会同时检查 3D 和 4D 输入，所以实现必须支持任意 leading dimensions：

```text
Q:    (..., queries, d_k)
K:    (..., keys, d_k)
V:    (..., keys, d_v)
mask: (..., queries, keys)
out:  (..., queries, d_v)
```

### 学生要做的事情

实现数值稳定 softmax，然后实现 attention：

```python
def softmax(x, dim):
    z = x - torch.max(x, dim=dim, keepdim=True).values
    exp = torch.exp(z)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    attn = softmax(scores, dim=-1)
    return attn @ V
```

### 相关答案

adapter：

```python
def run_scaled_dot_product_attention(Q, K, V, mask=None):
    from cs336_basics.model import scaled_dot_product_attention
    return scaled_dot_product_attention(Q, K, V, mask)
```

常见错误：

```python
# 错误：转置了最后一维以外的维度
K.T

# 错误：mask 语义反了。测试里 True 表示可以看，False 表示屏蔽
scores = scores.masked_fill(mask, -float("inf"))

# 错误：除以 sqrt(d_model)，多头里应该除以每个 head 的 d_k
scores = scores / math.sqrt(Q.shape[-1])
```

最后一条代码形式是对的，关键是进入 attention 的 Q 最后一维应该已经是 head_dim。

## 8. 第五步：RoPE

### 问题背景

RoPE 旋转位置编码作用在 Q 和 K 上，不作用在 V 上。它把最后一维按偶数/奇数两两成对旋转：

```text
x_even' = x_even * cos - x_odd * sin
x_odd'  = x_even * sin + x_odd * cos
```

频率：

```python
inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2) / d_k))
angles = position * inv_freq
```

### 学生要做的事情

实现一个模块，初始化时预计算 `cos` 和 `sin`：

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        assert d_k % 2 == 0
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(max_seq_len, device=device).float()
        angles = positions[:, None] * inv_freq[None, :]
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x, token_positions):
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        while cos.ndim < x.ndim - 1:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        y_even = x_even * cos - x_odd * sin
        y_odd = x_even * sin + x_odd * cos
        return torch.stack((y_even, y_odd), dim=-1).flatten(-2)
```

### 相关答案

adapter：

```python
def run_rope(d_k, theta, max_seq_len, in_query_or_key, token_positions):
    from cs336_basics.model import RotaryPositionalEmbedding
    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=in_query_or_key.device)
    return rope(in_query_or_key, token_positions)
```

常见错误：

```python
# 错误：把前半维和后半维配对，而不是偶数/奇数配对
x1, x2 = x[..., : d_k // 2], x[..., d_k // 2 :]

# 错误：RoPE 用 d_model 初始化。多头 attention 里应该用 head_dim
RotaryPositionalEmbedding(theta, d_model, max_seq_len)

# 错误：token_positions shape 是 (batch, seq)，要能 broadcast 到 (..., heads, seq, d)
```

## 9. 第六步：Multi-Head Self-Attention

### 问题背景

多头自注意力的输入输出 shape：

```text
x:      (..., seq, d_model)
q/k/v:  (..., heads, seq, head_dim)
out:    (..., seq, d_model)
```

测试要求 Q/K/V 投影权重都是 `(d_model, d_model)`，并且一次矩阵乘法得到所有 head，再 reshape。

### 学生要做的事情

实现：

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=None, theta=None, use_rope=False, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len, device=device) if use_rope else None

    def forward(self, x, token_positions=None):
        q = rearrange(self.q_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)
        v = rearrange(self.v_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)

        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        out = scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, "... h s d -> ... s (h d)")
        return self.output_proj(out)
```

### 相关答案

adapter：

```python
def run_multihead_self_attention(
    d_model, num_heads, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features
):
    from cs336_basics.model import MultiHeadSelfAttention
    layer = MultiHeadSelfAttention(d_model, num_heads, device=in_features.device, dtype=in_features.dtype)
    layer.q_proj.weight.data.copy_(q_proj_weight)
    layer.k_proj.weight.data.copy_(k_proj_weight)
    layer.v_proj.weight.data.copy_(v_proj_weight)
    layer.output_proj.weight.data.copy_(o_proj_weight)
    return layer(in_features)
```

带 RoPE 的 adapter：

```python
def run_multihead_self_attention_with_rope(
    d_model, num_heads, max_seq_len, theta,
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
    in_features, token_positions=None,
):
    from cs336_basics.model import MultiHeadSelfAttention
    layer = MultiHeadSelfAttention(
        d_model, num_heads, max_seq_len=max_seq_len, theta=theta,
        use_rope=True, device=in_features.device, dtype=in_features.dtype
    )
    layer.q_proj.weight.data.copy_(q_proj_weight)
    layer.k_proj.weight.data.copy_(k_proj_weight)
    layer.v_proj.weight.data.copy_(v_proj_weight)
    layer.output_proj.weight.data.copy_(o_proj_weight)
    return layer(in_features, token_positions=token_positions)
```

常见错误：

```python
# 错误：RoPE 作用到 V
v = self.rope(v, token_positions)

# 错误：没有 causal mask，TransformerLM 会看到未来 token
out = scaled_dot_product_attention(q, k, v)

# 错误：heads 和 seq 维度排反
rearrange(q, "... s (h d) -> ... s h d", h=num_heads)
```

## 10. 第七步：TransformerBlock

### 问题背景

本作业使用 pre-norm decoder-only Transformer block。顺序是：

```text
x = x + attention(RMSNorm(x))
x = x + ffn(RMSNorm(x))
```

注意第二个 RMSNorm 的输入是第一个 residual 之后的 `x`。

### 学生要做的事情

实现：

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, device=None, dtype=None):
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, max_seq_len=max_seq_len, theta=theta,
            use_rope=True, device=device, dtype=dtype
        )
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x
```

### 相关答案

`test_transformer_block` 给 adapter 的 `weights` key 已经去掉了 `layers.0.` 前缀：

```text
attn.q_proj.weight
attn.k_proj.weight
attn.v_proj.weight
attn.output_proj.weight
ln1.weight
ffn.w1.weight
ffn.w2.weight
ffn.w3.weight
ln2.weight
```

因此如果你的模块属性名刚好是 `attn`、`ln1`、`ffn`、`ln2`，可以直接：

```python
layer.load_state_dict(weights)
```

adapter：

```python
def run_transformer_block(d_model, num_heads, d_ff, max_seq_len, theta, weights, in_features):
    from cs336_basics.model import TransformerBlock
    layer = TransformerBlock(
        d_model, num_heads, d_ff, max_seq_len, theta,
        device=in_features.device, dtype=in_features.dtype
    )
    layer.load_state_dict(weights)
    token_positions = torch.arange(in_features.shape[-2], device=in_features.device)
    token_positions = token_positions.expand(in_features.shape[0], -1)
    return layer(in_features, token_positions=token_positions)
```

常见错误：

```python
# 错误：post-norm
x = self.ln1(x + self.attn(x))

# 错误：第二个 residual 加回了原始输入，而不是 attention 之后的 x
z = x + self.attn(self.ln1(x))
return x + self.ffn(self.ln2(z))
```

## 11. 第八步：TransformerLM

### 问题背景

完整 LM 的结构是：

```text
token ids
  -> token_embeddings
  -> TransformerBlock x num_layers
  -> ln_final
  -> lm_head
  -> logits
```

输出是未归一化 logits，不要在模型里 softmax。

### 学生要做的事情

实现：

```python
class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices):
        x = self.token_embeddings(in_indices)
        token_positions = torch.arange(in_indices.shape[-1], device=in_indices.device)
        token_positions = token_positions.expand(in_indices.shape[0], -1)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        return self.lm_head(self.ln_final(x))
```

### 相关答案

`test_transformer_lm` 传入的 state dict key 是：

```text
token_embeddings.weight
layers.0.attn.q_proj.weight
layers.0.attn.k_proj.weight
...
layers.2.ln2.weight
ln_final.weight
lm_head.weight
```

所以只要你的属性命名一致，adapter 可以很短：

```python
def run_transformer_lm(
    vocab_size, context_length, d_model, num_layers, num_heads,
    d_ff, rope_theta, weights, in_indices
):
    from cs336_basics.model import TransformerLM
    model = TransformerLM(
        vocab_size, context_length, d_model, num_layers, num_heads,
        d_ff, rope_theta, device=in_indices.device
    )
    model.load_state_dict(weights)
    return model(in_indices)
```

常见错误：

```python
# 错误：返回概率。测试期望 logits
return torch.softmax(self.lm_head(x), dim=-1)

# 错误：token_positions 固定用 context_length，截断输入测试会失败
token_positions = torch.arange(self.context_length)

# 错误：忘了 final norm
return self.lm_head(x)
```

`test_transformer_lm_truncated_input` 会把输入序列截成一半，所以 forward 里位置张量必须按当前输入长度生成，而不是永远生成 `context_length`。

## 12. Adapter 总模板

### 问题背景

adapter 是最容易“看起来写完了但测试不动”的地方。它必须和你的模块属性名一致。

### 学生要做的事情

如果你按本文推荐命名实现 `model.py`，可以把 Transformer 相关 adapter 写成下面这种模式。

### 相关答案

```python
def run_swiglu(d_model, d_ff, w1_weight, w2_weight, w3_weight, in_features):
    from cs336_basics.model import SwiGLU
    layer = SwiGLU(d_model, d_ff, device=in_features.device, dtype=in_features.dtype)
    layer.w1.weight.data.copy_(w1_weight)
    layer.w2.weight.data.copy_(w2_weight)
    layer.w3.weight.data.copy_(w3_weight)
    return layer(in_features)


def run_scaled_dot_product_attention(Q, K, V, mask=None):
    from cs336_basics.model import scaled_dot_product_attention
    return scaled_dot_product_attention(Q, K, V, mask)


def run_transformer_block(d_model, num_heads, d_ff, max_seq_len, theta, weights, in_features):
    from cs336_basics.model import TransformerBlock
    layer = TransformerBlock(
        d_model, num_heads, d_ff, max_seq_len, theta,
        device=in_features.device, dtype=in_features.dtype
    )
    layer.load_state_dict(weights)
    token_positions = torch.arange(in_features.shape[-2], device=in_features.device)
    token_positions = token_positions.expand(in_features.shape[0], -1)
    return layer(in_features, token_positions=token_positions)
```

如果 `load_state_dict` 报 missing keys / unexpected keys，优先检查你的属性名：

```text
推荐属性名                 对应 state_dict key
token_embeddings.weight     token_embeddings.weight
layers.0.attn.q_proj.weight layers.0.attn.q_proj.weight
layers.0.ffn.w1.weight      layers.0.ffn.w1.weight
ln_final.weight             ln_final.weight
lm_head.weight              lm_head.weight
```

## 13. 调试路线图

### 问题背景

Transformer 的错误通常不是“公式完全不会”，而是一个 shape、一次转置、一个 mask 语义、一个 module key 没对齐。按依赖顺序调试会省很多时间。

### 学生要做的事情

推荐路线：

```text
Linear / Embedding
  -> RMSNorm
  -> SiLU / SwiGLU
  -> scaled_dot_product_attention
  -> RoPE
  -> MultiHeadSelfAttention
  -> MultiHeadSelfAttention with RoPE
  -> TransformerBlock
  -> TransformerLM
```

每过一层就跑对应测试，不要等完整 LM 才一起跑。

### 相关答案

高频自查清单：

```text
Linear 是否是 x @ W.T？
Embedding 是否直接 weight[token_ids]？
RMSNorm 是否只在最后一维归一化，且不减均值？
SwiGLU 的 W2 是否是 d_ff -> d_model？
Attention mask 是否 True=保留，False=屏蔽？
Attention 是否支持 4D 输入？
RoPE 是否按偶数/奇数维配对？
RoPE 是否只作用 Q/K，不作用 V？
MHA reshape 是否是 (..., heads, seq, head_dim)？
TransformerBlock 是否是 pre-norm？
TransformerLM 是否返回 logits，不做 softmax？
截断输入时 token_positions 是否按当前 seq_len 生成？
```

出现 snapshot 不一致时，建议先打印这些 shape：

```python
print("x", x.shape)
print("q", q.shape, "k", k.shape, "v", v.shape)
print("mask", mask.shape, mask.dtype)
print("out", out.shape)
```

不要长期保留这些打印；测试通过后删掉。

## 14. 当前仓库状态下的特别提醒

### 问题背景

当前 `cs336_basics/model.py` 里有一段较完整的参考思路被注释掉了，但实际类还没有全部启用。`tests/adapters.py` 中 `run_linear` 和 `run_embedding` 已经接了一部分实现，但写法依赖当前的包装属性，例如 `linear.linear.weight` 和 `embedding.weight.weight`。

### 学生要做的事情

如果你决定按本文推荐方式重写 `Linear` / `Embedding`，记得同步更新 adapter。否则测试会在属性访问处失败。

### 相关答案

两种风格不要混用：

```python
# 风格 A：推荐
class Linear(nn.Module):
    self.weight = nn.Parameter(...)

layer.weight.data.copy_(weights)
```

```python
# 风格 B：当前雏形
class Linear(nn.Module):
    self.linear = nn.Linear(...)

layer.linear.weight.data.copy_(weights)
```

作业语义上更接近风格 A。风格 B 可能能让个别测试通过，但不符合“从零实现”的要求，也会让后面的 state dict key 更别扭。

## 15. 最小验收命令

### 问题背景

模型部分写完后，至少要单独验收 `tests/test_model.py`。不要只跑完整 `pytest`，因为别的部分失败会掩盖模型部分的真实状态。

### 学生要做的事情

按顺序运行：

```bash
.venv\Scripts\python.exe -m pytest tests/test_model.py -k "linear or embedding" -q
.venv\Scripts\python.exe -m pytest tests/test_model.py -k "rmsnorm or silu or swiglu" -q
.venv\Scripts\python.exe -m pytest tests/test_model.py -k "scaled_dot_product_attention or rope" -q
.venv\Scripts\python.exe -m pytest tests/test_model.py -k "multihead" -q
.venv\Scripts\python.exe -m pytest tests/test_model.py -k "transformer" -q
.venv\Scripts\python.exe -m pytest tests/test_model.py -q
```

### 相关答案

全部通过时，你应该看到类似：

```text
12 passed
```

如果只剩 `test_transformer_lm` 或 `test_transformer_block` 失败，通常优先查：

```text
load_state_dict key 是否匹配
pre-norm 顺序是否正确
causal mask 是否存在
RoPE 是否传入 token_positions
final RMSNorm 是否存在
lm_head 是否返回 logits
```

