from __future__ import annotations

from collections.abc import Iterable
import math
import os
from typing import IO, BinaryIO

import numpy as np
import torch


def softmax(in_features, dim):
    z = in_features - torch.max(in_features, dim=dim, keepdim=True).values
    exp = torch.exp(z)
    return exp / torch.sum(exp, dim=dim, keepdim=True)
def cross_entropy(inputs, targets):
    MAX=torch.max(inputs,dim=-1,keepdim=True).values
    shifted=inputs-MAX
    ALL=torch.log(torch.sum(torch.exp(shifted),dim=-1))+MAX.squeeze(-1)
    gold=inputs[torch.arange(inputs.shape[0]), targets]
    return torch.mean(ALL-gold)

def gradient_clipping(parameters, max_l2_norm):
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return

    total_norm = torch.sqrt(sum(torch.sum(p.grad.detach() ** 2) for p in params))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for p in params:
            p.grad.mul_(scale)
#
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
def get_batch(dataset, batch_size, context_length, device):
    starts = torch.randint(0, len(dataset) - context_length, (batch_size,)).numpy()
    x_np = np.stack([dataset[i : i + context_length] for i in starts]).copy()
    y_np = np.stack([dataset[i + 1 : i + context_length + 1] for i in starts]).copy()
    x = torch.as_tensor(x_np, dtype=torch.long, device=device)
    y = torch.as_tensor(y_np, dtype=torch.long, device=device)
    return x, y
    ...
def save_checkpoint(model, optimizer, iteration, out):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )
def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])
