# CS336 Assignment 1 Optimizer、Loss、Data、Checkpoint 实现专题文档

本文只讲 Assignment 1 第四部分：`softmax`、`cross_entropy`、`gradient_clipping`、`AdamW`、cosine learning-rate schedule、`get_batch`、`save_checkpoint`、`load_checkpoint`，以及它们如何接到 `tests/adapters.py` 并通过 `tests/test_nn_utils.py`、`tests/test_optimizer.py`、`tests/test_data.py`、`tests/test_serialization.py`。

本地当前状态备注：`tests/adapters.py` 中 `run_softmax` 已经接到了 `cs336_basics.model.softmax`，但 `run_cross_entropy`、`run_gradient_clipping`、`get_adamw_cls`、`run_get_lr_cosine_schedule`、`run_get_batch`、`run_save_checkpoint`、`run_load_checkpoint` 仍然是 `NotImplementedError`。当前仓库还没有专门的 `training.py` / `optimizer.py`，建议新建一个实现文件，例如 `cs336_basics/training.py`，adapter 只负责转发。

## 1. 测试架构

### 问题背景

CS336 的这部分测试同样只认识 `tests/adapters.py` 里的入口：

```python
run_softmax(in_features, dim)
run_cross_entropy(inputs, targets)
run_gradient_clipping(parameters, max_l2_norm)
get_adamw_cls()
run_get_lr_cosine_schedule(...)
run_get_batch(dataset, batch_size, context_length, device)
run_save_checkpoint(model, optimizer, iteration, out)
run_load_checkpoint(src, model, optimizer)
```

调用链建议整理成：

```text
tests/test_nn_utils.py
  -> tests/adapters.py::run_softmax
    -> cs336_basics/training.py::softmax

tests/test_optimizer.py
  -> tests/adapters.py::get_adamw_cls
    -> cs336_basics/training.py::AdamW

tests/test_data.py
  -> tests/adapters.py::run_get_batch
    -> cs336_basics/training.py::get_batch

tests/test_serialization.py
  -> tests/adapters.py::run_save_checkpoint / run_load_checkpoint
    -> cs336_basics/training.py::save_checkpoint / load_checkpoint
```

### 学生要做的事情

推荐新建：

```text
cs336_basics/training.py
```

把第四部分所有函数先放进去。等作业变大后，也可以拆成：

```text
cs336_basics/nn_utils.py
cs336_basics/optimizer.py
cs336_basics/data.py
cs336_basics/serialization.py
```

但在这个作业里，一个 `training.py` 最省事。

### 相关答案

adapter 只做胶水，例如：

```python
def run_cross_entropy(inputs, targets):
    from cs336_basics.training import cross_entropy
    return cross_entropy(inputs, targets)


def get_adamw_cls():
    from cs336_basics.training import AdamW
    return AdamW
```

不要把 AdamW、loss 或 checkpoint 的主体逻辑写在 `tests/adapters.py` 里。

## 2. 相关测试文件讲解

### 问题背景

这部分对应四个测试文件：

```text
tests/test_nn_utils.py          # softmax、cross_entropy、gradient clipping
tests/test_optimizer.py         # AdamW、cosine LR schedule
tests/test_data.py              # get_batch
tests/test_serialization.py     # checkpoint save/load
```

### 学生要做的事情

先分模块跑，不要直接全量跑：

```bash
.venv\Scripts\python.exe -m pytest tests/test_nn_utils.py -q
.venv\Scripts\python.exe -m pytest tests/test_optimizer.py -q
.venv\Scripts\python.exe -m pytest tests/test_data.py -q
.venv\Scripts\python.exe -m pytest tests/test_serialization.py -q
```

如果 `uv` 可用：

```bash
uv run pytest tests/test_nn_utils.py tests/test_optimizer.py tests/test_data.py tests/test_serialization.py -q
```

### 相关答案

测试重点：

```text
softmax:             必须数值稳定，x + 100 后输出不变
cross_entropy:       必须数值稳定，1000 * logits 不 overflow
gradient_clipping:   必须和 torch.nn.utils.clip_grad_norm_ 行为匹配
AdamW:               1000 步优化后权重要匹配 PyTorch AdamW 或参考 snapshot
LR schedule:         25 个指定 iteration 的学习率要精确匹配
get_batch:           x/y shape 正确，y = x + 1，起点随机且覆盖合法范围
checkpoint:          model state、optimizer state、iteration 都要恢复一致
```

## 3. 文件和函数规划

### 问题背景

第四部分可以按“训练工具箱”来组织。它们会在后续训练循环里复用。

### 学生要做的事情

推荐 `training.py` 结构：

```python
from __future__ import annotations

from collections.abc import Iterable
import math
import os
from typing import IO, BinaryIO

import numpy as np
import torch


def softmax(in_features, dim): ...
def cross_entropy(inputs, targets): ...
def gradient_clipping(parameters, max_l2_norm): ...

class AdamW(torch.optim.Optimizer): ...

def get_lr_cosine_schedule(...): ...
def get_batch(dataset, batch_size, context_length, device): ...
def save_checkpoint(model, optimizer, iteration, out): ...
def load_checkpoint(src, model, optimizer): ...
```

### 相关答案

依赖关系：

```text
cross_entropy
  -> logsumexp trick

AdamW
  -> torch.optim.Optimizer base class
  -> optimizer.state_dict / load_state_dict

save_checkpoint / load_checkpoint
  -> model.state_dict
  -> optimizer.state_dict
  -> torch.save / torch.load
```

AdamW 一定要继承 `torch.optim.Optimizer`，否则 checkpoint 测试里的 optimizer state dict 比较会很麻烦。

## 4. 第一步：数值稳定 Softmax

### 问题背景

普通 softmax：

```text
softmax(x_i) = exp(x_i) / sum_j exp(x_j)
```

直接 `torch.exp(x)` 会在大 logit 时 overflow。稳定写法是在指定维度上减去最大值：

```text
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
```

### 学生要做的事情

实现：

```python
def softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    z = in_features - torch.max(in_features, dim=dim, keepdim=True).values
    exp = torch.exp(z)
    return exp / torch.sum(exp, dim=dim, keepdim=True)
```

### 相关答案

adapter：

```python
def run_softmax(in_features, dim):
    from cs336_basics.training import softmax
    return softmax(in_features, dim)
```

如果你已经在 `cs336_basics/model.py` 里写了 `softmax`，也可以继续从 `model` import。但为了后续训练工具集中，推荐迁到 `training.py` 或在 `training.py` 中复用同一个实现。

常见错误：

```python
# 错误：没有 keepdim，broadcast 容易错
z = x - torch.max(x, dim=dim).values

# 错误：没有减 max，大数测试会 overflow
return torch.exp(x) / torch.exp(x).sum(dim=dim, keepdim=True)
```

## 5. 第二步：Cross Entropy

### 问题背景

测试传入的是二维 logits：

```text
inputs:  (batch_size, vocab_size)
targets: (batch_size,)
```

目标是平均负 log likelihood：

```text
loss = mean(logsumexp(logits_i) - logits_i[target_i])
```

不要先 softmax 再 log。那样更慢，也更容易数值不稳定。

### 学生要做的事情

实现稳定版：

```python
def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    maxes = torch.max(inputs, dim=-1, keepdim=True).values
    shifted = inputs - maxes
    logsumexp = torch.log(torch.sum(torch.exp(shifted), dim=-1)) + maxes.squeeze(-1)
    gold = inputs[torch.arange(inputs.shape[0], device=inputs.device), targets]
    return torch.mean(logsumexp - gold)
```

### 相关答案

adapter：

```python
def run_cross_entropy(inputs, targets):
    from cs336_basics.training import cross_entropy
    return cross_entropy(inputs, targets)
```

语言模型训练时通常会先 reshape：

```python
loss = cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
```

常见错误：

```python
# 错误：先 softmax 再 log，数值不稳
probs = softmax(inputs, dim=-1)
loss = -torch.log(probs[torch.arange(inputs.shape[0]), targets]).mean()

# 错误：targets 是类别 id，不是 one-hot
gold = torch.sum(inputs * targets, dim=-1)
```

## 6. 第三步：Gradient Clipping

### 问题背景

测试希望你的实现和 PyTorch 的 `clip_grad_norm_` 匹配。它计算所有非空梯度的全局 L2 norm，如果超过 `max_l2_norm`，就按同一个系数原地缩放所有梯度。

公式：

```text
total_norm = sqrt(sum_p sum(p.grad^2))
scale = max_l2_norm / (total_norm + eps)
if scale < 1: grad *= scale
```

### 学生要做的事情

实现：

```python
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return

    total_norm = torch.sqrt(sum(torch.sum(p.grad.detach() ** 2) for p in params))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for p in params:
            p.grad.mul_(scale)
```

### 相关答案

adapter：

```python
def run_gradient_clipping(parameters, max_l2_norm):
    from cs336_basics.training import gradient_clipping
    return gradient_clipping(parameters, max_l2_norm)
```

常见错误：

```python
# 错误：逐个参数分别裁剪，而不是全局 norm
for p in params:
    p.grad *= max_l2_norm / p.grad.norm()

# 错误：修改 parameter 本身，而不是 parameter.grad
p.mul_(scale)

# 错误：没有跳过 grad is None 的参数
```

## 7. 第四步：AdamW

### 问题背景

AdamW 是 Adam 加 decoupled weight decay。测试会用你的 optimizer 跑 1000 步小模型，然后和 PyTorch AdamW 或参考 snapshot 比较。

核心状态：

```text
step
exp_avg       # m
exp_avg_sq    # v
```

更新公式常见写法：

```text
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad^2
step_size = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
param -= step_size * m / (sqrt(v) + eps)
param -= lr * weight_decay * param
```

### 学生要做的事情

实现一个继承 `torch.optim.Optimizer` 的类：

```python
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                step_size = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.addcdiv_(m, torch.sqrt(v).add(eps), value=-step_size)

                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)

        return loss
```

### 相关答案

adapter：

```python
def get_adamw_cls():
    from cs336_basics.training import AdamW
    return AdamW
```

常见错误：

```python
# 错误：没有继承 torch.optim.Optimizer
class AdamW:
    ...

# 错误：没有 @torch.no_grad()，step 会进入计算图
def step(self):
    ...

# 错误：把 weight decay 加到 grad 里。这是 Adam，不是 decoupled AdamW
grad = grad + weight_decay * p

# 错误：没有 bias correction
p.addcdiv_(m, torch.sqrt(v).add(eps), value=-lr)
```

测试允许和 PyTorch AdamW 或课程参考实现任意一个匹配，因为 decoupled weight decay 的浮点细节可能有微小差异。

## 8. 第五步：Cosine Learning-Rate Schedule

### 问题背景

学习率分三段：

```text
1. it < warmup_iters:
   线性 warmup，从 0 到 max_learning_rate

2. warmup_iters <= it <= cosine_cycle_iters:
   cosine decay，从 max_learning_rate 到 min_learning_rate

3. it > cosine_cycle_iters:
   固定为 min_learning_rate
```

### 学生要做的事情

实现：

```python
def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate

    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)
```

### 相关答案

adapter：

```python
def run_get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    from cs336_basics.training import get_lr_cosine_schedule
    return get_lr_cosine_schedule(
        it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters
    )
```

测试里的关键边界：

```text
it = 0                 -> 0
it = warmup_iters      -> max_learning_rate
it = cosine_cycle_iters -> min_learning_rate
it > cosine_cycle_iters -> min_learning_rate
```

常见错误：

```python
# 错误：warmup 结束点少一格，导致 it=warmup_iters 时不是 max_lr
return max_lr * it / (warmup_iters - 1)

# 错误：cosine progress 分母写成 cosine_cycle_iters
progress = (it - warmup_iters) / cosine_cycle_iters
```

## 9. 第六步：get_batch

### 问题背景

语言模型 batch 从一维 token id 数组中随机采样连续片段：

```text
x = dataset[start : start + context_length]
y = dataset[start + 1 : start + context_length + 1]
```

测试会检查：

```text
x.shape == (batch_size, context_length)
y.shape == (batch_size, context_length)
y == x + 1       # 在 dataset=np.arange(100) 的测试里成立
start 覆盖 0 到 len(dataset) - context_length - 1
device 参数真的生效
```

### 学生要做的事情

实现：

```python
def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str):
    starts = torch.randint(0, len(dataset) - context_length, (batch_size,))
    x = torch.stack([
        torch.as_tensor(dataset[i : i + context_length], dtype=torch.long)
        for i in starts
    ])
    y = torch.stack([
        torch.as_tensor(dataset[i + 1 : i + context_length + 1], dtype=torch.long)
        for i in starts
    ])
    return x.to(device), y.to(device)
```

### 相关答案

adapter：

```python
def run_get_batch(dataset, batch_size, context_length, device):
    from cs336_basics.training import get_batch
    return get_batch(dataset, batch_size, context_length, device)
```

为什么 `torch.randint(0, len(dataset) - context_length, ...)` 是对的？

```text
high 是 exclusive
最大 start = len(dataset) - context_length - 1
y 最后访问 start + context_length
当 start 最大时，y 最后访问 len(dataset) - 1，刚好合法
```

常见错误：

```python
# 错误：high 多 1，可能采到非法 start
torch.randint(0, len(dataset) - context_length + 1, ...)

# 错误：y 没有右移
y = dataset[start : start + context_length]

# 错误：忽略 device，cuda:99 测试不会抛出预期错误
return x, y
```

## 10. 第七步：Checkpoint 保存

### 问题背景

checkpoint 至少要保存三样：

```text
model.state_dict()
optimizer.state_dict()
iteration
```

测试会先训练 10 步小网络，保存，再加载到新模型和新 optimizer，最后比较 model state、optimizer state 和 iteration 是否完全恢复。

### 学生要做的事情

实现：

```python
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )
```

### 相关答案

adapter：

```python
def run_save_checkpoint(model, optimizer, iteration, out):
    from cs336_basics.training import save_checkpoint
    return save_checkpoint(model, optimizer, iteration, out)
```

常见错误：

```python
# 错误：保存整个 model 对象，跨代码版本更脆弱
torch.save(model, out)

# 错误：忘了 optimizer，恢复训练时 AdamW 动量状态丢失
torch.save({"model": model.state_dict(), "iteration": iteration}, out)

# 错误：key 名和 load_checkpoint 不一致
```

## 11. 第八步：Checkpoint 加载

### 问题背景

加载时要把 checkpoint 中的 state 恢复到调用者传入的模型和 optimizer 上，并返回 iteration。

### 学生要做的事情

实现：

```python
def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])
```

### 相关答案

adapter：

```python
def run_load_checkpoint(src, model, optimizer):
    from cs336_basics.training import load_checkpoint
    return load_checkpoint(src, model, optimizer)
```

常见错误：

```python
# 错误：创建并返回一个新 model。测试希望修改传入的 model
model = MyModel()
model.load_state_dict(...)
return model

# 错误：忘了 load optimizer
model.load_state_dict(checkpoint["model"])
return checkpoint["iteration"]

# 错误：不返回 iteration
```

## 12. Adapter 总模板

### 问题背景

如果实现放在 `cs336_basics/training.py`，`tests/adapters.py` 这部分可以非常薄。

### 学生要做的事情

把下面模式移植到对应函数中。注意不要改函数签名。

### 相关答案

```python
def run_get_batch(dataset, batch_size, context_length, device):
    from cs336_basics.training import get_batch
    return get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features, dim):
    from cs336_basics.training import softmax
    return softmax(in_features, dim)


def run_cross_entropy(inputs, targets):
    from cs336_basics.training import cross_entropy
    return cross_entropy(inputs, targets)


def run_gradient_clipping(parameters, max_l2_norm):
    from cs336_basics.training import gradient_clipping
    return gradient_clipping(parameters, max_l2_norm)


def get_adamw_cls():
    from cs336_basics.training import AdamW
    return AdamW


def run_get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    from cs336_basics.training import get_lr_cosine_schedule
    return get_lr_cosine_schedule(
        it=it,
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
    )


def run_save_checkpoint(model, optimizer, iteration, out):
    from cs336_basics.training import save_checkpoint
    return save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(src, model, optimizer):
    from cs336_basics.training import load_checkpoint
    return load_checkpoint(src, model, optimizer)
```

## 13. 最小完整实现参考

### 问题背景

这一段给出一个集中在 `training.py` 的最小实现骨架。你可以直接照着结构写，但要根据自己的文件拆分调整 import。

### 学生要做的事情

新建 `cs336_basics/training.py`，放入：

```python
from __future__ import annotations

from collections.abc import Iterable
import math
import os
from typing import IO, BinaryIO

import numpy as np
import torch


def softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    z = in_features - torch.max(in_features, dim=dim, keepdim=True).values
    exp = torch.exp(z)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    maxes = torch.max(inputs, dim=-1, keepdim=True).values
    shifted = inputs - maxes
    logsumexp = torch.log(torch.sum(torch.exp(shifted), dim=-1)) + maxes.squeeze(-1)
    gold = inputs[torch.arange(inputs.shape[0], device=inputs.device), targets]
    return torch.mean(logsumexp - gold)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return
    total_norm = torch.sqrt(sum(torch.sum(p.grad.detach() ** 2) for p in params))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for p in params:
            p.grad.mul_(scale)


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                step_size = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.addcdiv_(m, torch.sqrt(v).add(eps), value=-step_size)

                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)
        return loss


def get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str):
    starts = torch.randint(0, len(dataset) - context_length, (batch_size,))
    x = torch.stack([
        torch.as_tensor(dataset[i : i + context_length], dtype=torch.long)
        for i in starts
    ])
    y = torch.stack([
        torch.as_tensor(dataset[i + 1 : i + context_length + 1], dtype=torch.long)
        for i in starts
    ])
    return x.to(device), y.to(device)


def save_checkpoint(model, optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model, optimizer) -> int:
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])
```

### 相关答案

如果你后续把 `softmax` 留在 `model.py`，也没问题，但建议避免重复实现两个不同版本。最简单办法是在 `model.py` 和 `training.py` 里只保留一个真实实现，另一个 import 复用。

## 14. 调试路线图

### 问题背景

这部分的失败大多来自边界条件：数值稳定、随机采样上界、optimizer state dict、device 是否生效。

### 学生要做的事情

按这个顺序调：

```text
softmax
  -> cross_entropy
  -> gradient_clipping
  -> AdamW
  -> LR schedule
  -> get_batch
  -> checkpoint
```

### 相关答案

高频自查清单：

```text
softmax 是否减了 max？
cross_entropy 是否用 logsumexp trick？
cross_entropy 的 targets 是否是一维类别 id？
gradient clipping 是否是全局 L2 norm？
AdamW 是否继承 torch.optim.Optimizer？
AdamW 是否保存 step、exp_avg、exp_avg_sq？
AdamW 是否 decoupled weight decay？
LR schedule 的 warmup 和 cosine 边界是否正确？
get_batch 的 randint high 是否是 len(dataset) - context_length？
get_batch 是否返回 torch.long？
get_batch 是否把 x/y 移到 device？
checkpoint 是否保存并加载 optimizer state？
load_checkpoint 是否返回 int iteration？
```

## 15. 最小验收命令

### 问题背景

这部分实现完后，先跑四个相关测试文件。它们和 BPE、Transformer Model 可以分开验收。

### 学生要做的事情

运行：

```bash
.venv\Scripts\python.exe -m pytest tests/test_nn_utils.py -q
.venv\Scripts\python.exe -m pytest tests/test_optimizer.py -q
.venv\Scripts\python.exe -m pytest tests/test_data.py -q
.venv\Scripts\python.exe -m pytest tests/test_serialization.py -q
```

全部一起：

```bash
.venv\Scripts\python.exe -m pytest tests/test_nn_utils.py tests/test_optimizer.py tests/test_data.py tests/test_serialization.py -q
```

### 相关答案

全部通过时，你应该看到这些测试都 passed。若失败，优先定位：

```text
test_softmax_matches_pytorch:     查 max trick 和 dim
test_cross_entropy:               查 logsumexp 和 target indexing
test_gradient_clipping:           查全局 norm 和 grad is None
test_adamw:                       查 bias correction、weight decay、state
test_get_lr_cosine_schedule:      查边界 it=warmup_iters / cosine_cycle_iters
test_get_batch:                   查随机起点 high、device、y 右移
test_checkpointing:               查 optimizer state 和 iteration
```

