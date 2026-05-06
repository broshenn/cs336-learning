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
    
#
# def get_lr_cosine_schedule(...): ...
# def get_batch(dataset, batch_size, context_length, device): ...
# def save_checkpoint(model, optimizer, iteration, out): ...
# def load_checkpoint(src, model, optimizer): ...