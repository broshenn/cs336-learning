# cs336_basics/model.py
from __future__ import annotations

import math
import torch
from einops import rearrange
from torch import nn

from tests.conftest import batch_size


class Linear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=False)
    def forward(self,in_features):
        return self.linear(in_features)

class Embedding(nn.Module):
    def __init__(self,num_embeddings, embedding_dim):
        super().__init__()
        self.weight=nn.Embedding(num_embeddings, embedding_dim)

    def forward(self,inputs):
        return self.weight(inputs)

class RMSNorm(nn.Module):
    def __init__(self,d_model,eps=1e-5):
        super().__init__()
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(d_model))

    def forward(self,x:torch.Tensor):
        rms=torch.sqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
        return x / rms*self.weight

def silu(x):
    return x * torch.sigmoid(x)
#
#
class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ff):
        super().__init__()
        self.w1=nn.Linear(d_model,d_ff,bias=False)
        self.w2=nn.Linear(d_ff,d_model,bias=False)
        self.w3=nn.Linear(d_model,d_ff,bias=False)

    def forward(self,x):
        return self.w2(silu(self.w1(x))*self.w3(x))

def softmax(x, dim):
    z=x-torch.max(x, dim=dim, keepdim=True).values
    exp = torch.exp(z)
    return exp/torch.sum(exp,dim=dim,keepdim=True)


def scaled_dot_product_attention(Q:torch.Tensor, K, V, mask=None):
    # [batch, head,seq_len, d_model]
    d_k=Q.shape[-1]
    scores=Q @ K.transpose(-2,-1)/math.sqrt(d_k)
    if mask is not None:
        scores=scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    attn=softmax(scores,-1)
    return attn @ V

def precompute_freqs_cis(d_k, seq_len: int,theta):
    freqs=1.0/theta**(torch.arange(0, d_k, 2)[:d_k//2]/d_k)
    t=torch.arange(seq_len)
    freqs=t.outer(freqs)
    return freqs.cos(),freqs.sin()

def rotate_half(x):
    x1=x[...,:x.shape[-1]//2]
    x2=x[...,x.shape[-1]//2:]
    return torch.cat((-x2,x1),dim=-1)

def apply_rotary_positional_embedding(x, cos, sin):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    y_even = x_even * cos - x_odd * sin
    y_odd = x_even * sin + x_odd * cos
    return torch.stack((y_even, y_odd), dim=-1).flatten(-2)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta=theta
        self.d_k=d_k
        self.max_seq_len=max_seq_len
        self.device=device
        self.cos,self.sin=precompute_freqs_cis(self.d_k,self.max_seq_len,self.theta)

    def forward(self, x, token_positions=None):
        seq_len = x.shape[-2]
        batch = x.shape[0]
        positions=token_positions
        cos = self.cos[positions]
        sin = self.sin[positions]
        return apply_rotary_positional_embedding(x, cos, sin)
#
#
# class RotaryPositionalEmbedding(nn.Module):
#     def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
#         super().__init__()
#         assert d_k % 2 == 0
#         inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
#         positions = torch.arange(max_seq_len, device=device).float()
#         angles = positions[:, None] * inv_freq[None, :]
#         self.register_buffer("cos", torch.cos(angles), persistent=False)
#         self.register_buffer("sin", torch.sin(angles), persistent=False)
#
#     def forward(self, x, token_positions=None):
#         seq_len = x.shape[-2]
#         if token_positions is None:
#             cos = self.cos[:seq_len]
#             sin = self.sin[:seq_len]
#             while cos.ndim < x.ndim - 1:
#                 cos = cos.unsqueeze(0)
#                 sin = sin.unsqueeze(0)
#         else:
#             cos = self.cos[token_positions]
#             sin = self.sin[token_positions]
#             while cos.ndim < x.ndim - 1:
#                 cos = cos.unsqueeze(-3)
#                 sin = sin.unsqueeze(-3)
#
#         x_even = x[..., 0::2]
#         x_odd = x[..., 1::2]
#         y_even = x_even * cos - x_odd * sin
#         y_odd = x_even * sin + x_odd * cos
#         return torch.stack((y_even, y_odd), dim=-1).flatten(-2)
#
#
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=None, theta=None, use_rope=False, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q=nn.Linear(d_model, d_model, bias=False)
        self.k=nn.Linear(d_model, d_model, bias=False)
        self.v=nn.Linear(d_model, d_model, bias=False)
        self.o=nn.Linear(d_model, d_model, bias=False)
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len, device)

    def forward(self, x, token_positions=None):
        batch_size,seq_len,d_model=x.shape
        q=self.q(x).view(batch_size, seq_len, self.num_heads,self.head_dim).transpose(1,2)
        k=self.k(x).view(batch_size, seq_len, self.num_heads,self.head_dim).transpose(1,2)
        v=self.v(x).view(batch_size, seq_len, self.num_heads,self.head_dim).transpose(1,2)
        if self.use_rope:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        attn_output = scaled_dot_product_attention(q, k, v,mask).transpose(1,2).reshape(batch_size, seq_len, d_model)
        return self.o(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, device=None, dtype=None):
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, max_seq_len=max_seq_len, theta=theta, use_rope=True, device=device, dtype=dtype
        )
        self.ln1 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln2 = RMSNorm(d_model)

    def forward(self,x, token_positions=None):
        batch_size,seq_len,d_model=x.shape
        if token_positions is None:
            token_positions=torch.arange(0, seq_len).expand(batch_size, seq_len)
        x=x+self.attn(self.ln1(x), token_positions=token_positions)
        x=x+self.ffn(self.ln2(x))
        return x

#
#
# class TransformerBlock(nn.Module):
#     def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, device=None, dtype=None):
#         super().__init__()
#         self.attn = MultiHeadSelfAttention(
#             d_model, num_heads, max_seq_len=max_seq_len, theta=theta, use_rope=True, device=device, dtype=dtype
#         )
#         self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
#         self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
#         self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
#
#     def forward(self, x, token_positions=None):
#         x = x + self.attn(self.ln1(x), token_positions=token_positions)
#         x = x + self.ffn(self.ln2(x))
#         return x
#
#
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
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, in_indices):
        x = self.token_embeddings(in_indices)
        token_positions = torch.arange(in_indices.shape[-1], device=in_indices.device)
        token_positions = token_positions.expand(in_indices.shape[0], -1)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        return self.lm_head(self.ln_final(x))