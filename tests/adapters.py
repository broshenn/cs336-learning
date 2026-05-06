from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    给定 Linear 层的权重，计算批量化输入的变换结果。

    参数:
        d_in (int): 输入维度大小
        d_out (int): 输出维度大小
        weights (Float[Tensor, "d_out d_in"]): 要使用的线性权重
        in_features (Float[Tensor, "... d_in"]): 要应用变换的输入张量

    返回:
        Float[Tensor, "... d_out"]: 线性模块的变换输出。
    """

    from cs336_basics.model import Linear
    linear=Linear(d_in,d_out)
    linear.linear.weight.data.copy_(weights)
    return linear(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    给定 Embedding 层的权重，根据一批 token id 获取对应的 embeddings。

    参数:
        vocab_size (int): 词表中的 embedding 数量
        d_model (int): embedding 维度大小
        weights (Float[Tensor, "vocab_size d_model"]): 要获取的 embedding 向量
        token_ids (Int[Tensor, "..."]): 要从 Embedding 层获取的 token id 集合

    返回:
        Float[Tensor, "... d_model"]: Embedding 层返回的批量 embeddings。
    """
    from cs336_basics.model import Embedding
    embedding=Embedding(vocab_size,d_model)
    embedding.weight.weight.data.copy_(weights)
    return embedding(token_ids)

def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """给定 SwiGLU 网络的权重，返回使用这些权重运行你实现的结果。

    参数:
        d_model (int): 前馈输入和输出的维度。
        d_ff (int): SwiGLU 内部上投影的维度。
        w1_weight (Float[Tensor, "d_ff d_model"]): W1 的存储权重
        w2_weight (Float[Tensor, "d_model d_ff"]): W2 的存储权重
        w3_weight (Float[Tensor, "d_ff d_model"]): W3 的存储权重
        in_features (Float[Tensor, "... d_model"]): 前馈层的输入 embeddings。

    返回:
        Float[Tensor, "... d_model"]: 与输入 embeddings 形状相同的输出 embeddings。
    """
    # 示例:
    # 如果你的 state dict keys 匹配，可以用 `load_state_dict()`
    from cs336_basics.model import SwiGLU
    swiglu=SwiGLU(d_model,d_ff)
    swiglu.w1.weight.data.copy_(w1_weight)
    swiglu.w2.weight.data.copy_(w2_weight)
    swiglu.w3.weight.data.copy_(w3_weight)
    return swiglu(in_features)
    # swiglu.load_state_dict(weights)
    # 也可以手动赋值权重
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    raise NotImplementedError


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    给定 key (K)、query (Q) 和 value (V) 张量，返回你的 scaled dot product attention 实现结果。

    参数:
        Q (Float[Tensor, " ... queries d_k"]): Query 张量
        K (Float[Tensor, " ... keys d_k"]): Key 张量
        V (Float[Tensor, " ... keys d_v"]): Value 张量
        mask (Bool[Tensor, " ... queries keys"] | None): Mask 张量

    返回:
        Float[Tensor, " ... queries d_v"]: SDPA 的输出
    """
    from cs336_basics.model import scaled_dot_product_attention
    return scaled_dot_product_attention(Q,K,V,mask)
    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    给定 naive 非批次实现的多头注意力机制的 key、query 和 value 投影权重，
    返回优化后的批次实现输出。此实现应该在单次矩阵乘法中处理所有 head 的 key、query 和 value 投影。
    此函数不应使用 RoPE。
    参见 Vaswani et al., 2017 的 3.2.2 节。

    参数:
        d_model (int): 前馈输入和输出的维度。
        num_heads (int): 多头注意力中使用的 head 数量。
        q_proj_weight (Float[Tensor, "d_model d_model"]): Q 投影的权重
        k_proj_weight (Float[Tensor, "d_model d_model"]): K 投影的权重
        v_proj_weight (Float[Tensor, "d_model d_model"]): V 投影的权重
        o_proj_weight (Float[Tensor, "d_model d_model"]): 输出投影的权重
        in_features (Float[Tensor, "... sequence_length d_model"]): 要运行实现的输入张量。

    返回:
        Float[Tensor, " ... sequence_length d_model"]: 使用给定 QKV 投影权重和输入特征
        运行优化批次多头注意力实现的输出张量。
    """
    from cs336_basics.model import MultiHeadSelfAttention
    multihead_self_attention=MultiHeadSelfAttention(d_model,num_heads)
    multihead_self_attention.q.weight.data.copy_(q_proj_weight)
    multihead_self_attention.k.weight.data.copy_(k_proj_weight)
    multihead_self_attention.v.weight.data.copy_(v_proj_weight)
    multihead_self_attention.o.weight.data.copy_(o_proj_weight)
    return multihead_self_attention(in_features)
    raise NotImplementedError


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    给定 naive 非批次实现的多头注意力机制的 key、query 和 value 投影权重，
    返回优化后的批次实现输出。此实现应该在单次矩阵乘法中处理所有 head 的 key、query 和 value 投影。
    此版本的 MHA 应该包含 RoPE。
    在这种情况下，RoPE embedding 维度必须是 head embedding 维度 (d_model // num_heads)。
    参见 Vaswani et al., 2017 的 3.2.2 节。

    参数:
        d_model (int): 前馈输入和输出的维度。
        num_heads (int): 多头注意力中使用的 head 数量。
        max_seq_len (int): 如果你的实现做预缓存，最大序列长度。
        theta (float): RoPE 参数。
        q_proj_weight (Float[Tensor, "d_model d_model"]): Q 投影的权重
        k_proj_weight (Float[Tensor, "d_model d_model"]): K 投影的权重
        v_proj_weight (Float[Tensor, "d_model d_model"]): V 投影的权重
        o_proj_weight (Float[Tensor, "d_model d_model"]): 输出投影的权重
        in_features (Float[Tensor, "... sequence_length d_model"]): 要运行实现的输入张量。
        token_positions (Int[Tensor, " ... sequence_length"] | None): 可选的 token 位置张量

    返回:
        Float[Tensor, " ... sequence_length d_model"]: 使用给定 QKV 投影权重和输入特征
        运行包含 RoPE 的优化批次多头注意力实现的输出张量。
    """
    from cs336_basics.model import MultiHeadSelfAttention
    multihead_self_attention=MultiHeadSelfAttention(d_model,num_heads,max_seq_len,theta,use_rope=True)
    multihead_self_attention.q.weight.data.copy_(q_proj_weight)
    multihead_self_attention.k.weight.data.copy_(k_proj_weight)
    multihead_self_attention.v.weight.data.copy_(v_proj_weight)
    multihead_self_attention.o.weight.data.copy_(o_proj_weight)
    return multihead_self_attention(in_features,token_positions)
    raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    对给定的输入张量运行 RoPE。

    参数:
        d_k (int): Query 或 key 张量的 embedding 维度大小。
        theta (float): RoPE 参数。
        max_seq_len (int): 如果你的实现做预缓存，最大序列长度。
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): 要运行 RoPE 的输入张量。
        token_positions (Int[Tensor, "... sequence_length"]): shape 为 (batch_size, sequence_length)
            的 token 位置张量

    返回:
        Float[Tensor, " ... sequence_length d_k"]: 应用 RoPE 后的输入张量。
    """
    from cs336_basics.model import RotaryPositionalEmbedding
    rope=RotaryPositionalEmbedding(theta, d_k, max_seq_len)
    return rope(in_query_or_key,token_positions)
    raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    给定 pre-norm Transformer block 的权重和输入特征，
    返回运行 Transformer block 的输出。

    此函数应使用 RoPE。
    根据你的实现，你可能只需要将相关参数传递给 TransformerBlock 构造函数，
    或者你需要初始化自己的 RoPE 类并传递它。

    参数:
        d_model (int): Transformer block 输入的维度。
        num_heads (int): 多头注意力中使用的 head 数量。`d_model` 必须能整除 `num_heads`。
        d_ff (int): 前馈内部层的维度。
        max_seq_len (int): 如果你的实现做预缓存，最大序列长度。
        theta (float): RoPE 参数。
        weights (dict[str, Tensor]):
            参考实现的 state dict。
            此字典的 key 包括:
            - `attn.q_proj.weight`
                所有 `num_heads` 注意力头的 query 投影。
                Shape 为 (d_model, d_model)。
                行按 shape 为 (num_heads, d_k) 的矩阵排序，
                所以 `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。
            - `attn.k_proj.weight`
                所有 `num_heads` 注意力头的 key 投影。
                Shape 为 (d_model, d_model)。
                行按 shape 为 (num_heads, d_k) 的矩阵排序，
                所以 `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。
            - `attn.v_proj.weight`
                所有 `num_heads` 注意力头的 value 投影。
                Shape 为 (d_model, d_model)。
                行按 shape 为 (num_heads, d_v) 的矩阵排序，
                所以 `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。
            - `attn.output_proj.weight`
                多头自注意力输出投影的权重。
                Shape 为 (d_model, d_model)。
            - `ln1.weight`
                Transformer block 中第一个 RMSNorm 的仿射变换权重。
                Shape 为 (d_model,)。
            - `ffn.w1.weight`
                FFN 中第一个线性变换的权重。
                Shape 为 (d_ff, d_model)。
            - `ffn.w2.weight`
                FFN 中第二个线性变换的权重。
                Shape 为 (d_model, d_ff)。
            - `ffn.w3.weight`
                FFN 中第三个线性变换的权重。
                Shape 为 (d_ff, d_model)。
            - `ln2.weight`
                Transformer block 中第二个 RMSNorm 的仿射变换权重。
                Shape 为 (d_model,)。
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            要运行实现的输入张量。

    返回:
        Float[Tensor, "batch sequence_length d_model"] 使用 RoPE 对输入特征
        运行 Transformer block 的输出张量。
    """
    from cs336_basics.model import TransformerBlock
    transformer_block=TransformerBlock(d_model,num_heads,d_ff,max_seq_len,theta)
    transformer_block.attn.q.weight.data.copy_(weights["attn.q_proj.weight"])
    transformer_block.attn.k.weight.data.copy_(weights["attn.k_proj.weight"])
    transformer_block.attn.v.weight.data.copy_(weights["attn.v_proj.weight"])
    transformer_block.attn.o.weight.data.copy_(weights["attn.output_proj.weight"])
    transformer_block.ln1.weight.data.copy_(weights["ln1.weight"])
    transformer_block.ffn.w1.weight.data.copy_(weights["ffn.w1.weight"])
    transformer_block.ffn.w2.weight.data.copy_(weights["ffn.w2.weight"])
    transformer_block.ffn.w3.weight.data.copy_(weights["ffn.w3.weight"])
    transformer_block.ln2.weight.data.copy_(weights["ln2.weight"])
    return transformer_block(in_features)
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """给定 Transformer 语言模型的权重和输入 indices，
    返回在前向传播上的输出。

    此函数应使用 RoPE。

    参数:
        vocab_size (int): 输出词表中要预测的唯一项数量。
        context_length (int): 每次最多处理的 token 数量。
        d_model (int): 模型 embeddings 和子层输出的维度。
        num_layers (int): 要使用的 Transformer 层数量。
        num_heads (int): 多头注意力中使用的 head 数量。`d_model` 必须能整除 `num_heads`。
        d_ff (int): 前馈内部层的维度（3.3 节）。
        rope_theta (float): RoPE $\\Theta$ 参数。
        weights (dict[str, Tensor]):
            参考实现的 state dict。{num_layers} 是指 0 到 num_layers - 1 之间的整数（层索引）。
            此字典的 key 包括:
            - `token_embeddings.weight`
                Token embedding 矩阵。Shape 为 (vocab_size, d_model)。
            - `layers.{num_layers}.attn.q_proj.weight`
                所有 `num_heads` 注意力头的 query 投影。
                Shape 为 (num_heads * (d_model / num_heads), d_model)。
                行按 shape 为 (num_heads, d_k) 的矩阵排序，
                所以 `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.k_proj.weight`
                所有 `num_heads` 注意力头的 key 投影。
                Shape 为 (num_heads * (d_model / num_heads), d_model)。
                行按 shape 为 (num_heads, d_k) 的矩阵排序，
                所以 `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.v_proj.weight`
                所有 `num_heads` 注意力头的 value 投影。
                Shape 为 (num_heads * (d_model / num_heads), d_model)。
                行按 shape 为 (num_heads, d_v) 的矩阵排序，
                所以 `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.output_proj.weight`
                多头自注意力输出投影的权重。
                Shape 为 ((d_model / num_heads) * num_heads, d_model)。
            - `layers.{num_layers}.ln1.weight`
                Transformer block 中第一个 RMSNorm 的仿射变换权重。
                Shape 为 (d_model,)。
            - `layers.{num_layers}.ffn.w1.weight`
                FFN 中第一个线性变换的权重。
                Shape 为 (d_ff, d_model)。
            - `layers.{num_layers}.ffn.w2.weight`
                FFN 中第二个线性变换的权重。
                Shape 为 (d_model, d_ff)。
            - `layers.{num_layers}.ffn.w3.weight`
                FFN 中第三个线性变换的权重。
                Shape 为 (d_ff, d_model)。
            - `layers.{num_layers}.ln2.weight`
                Transformer block 中第二个 RMSNorm 的仿射变换权重。
                Shape 为 (d_model,)。
            - `ln_final.weight`
                最终 transformer block 输出 RMSNorm 的仿射变换权重。
                Shape 为 (d_model, )。
            - `lm_head.weight`
                语言模型输出 embedding 的权重。
                Shape 为 (vocab_size, d_model)。
        in_indices (Int[Tensor, "batch_size sequence_length"]): 要运行语言模型的输入 indices 张量。
            Shape 为 (batch_size, sequence_length)，其中 `sequence_length` 最多为 `context_length`。

    返回:
        Float[Tensor, "batch_size sequence_length vocab_size"]: 包含每个 token 的
        预测非归一化下一个词分布的张量。
    """
    from cs336_basics.model import TransformerLM
    transformer_lm = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)

    # token embeddings
    transformer_lm.token_embeddings.weight.weight.data.copy_(weights["token_embeddings.weight"])

    # each layer
    for i in range(num_layers):
        layer_prefix = f"layers.{i}."
        attn = transformer_lm.layers[i].attn
        ffn = transformer_lm.layers[i].ffn
        ln1 = transformer_lm.layers[i].ln1
        ln2 = transformer_lm.layers[i].ln2

        # attention projections
        attn.q.weight.data.copy_(weights[f"{layer_prefix}attn.q_proj.weight"])
        attn.k.weight.data.copy_(weights[f"{layer_prefix}attn.k_proj.weight"])
        attn.v.weight.data.copy_(weights[f"{layer_prefix}attn.v_proj.weight"])
        attn.o.weight.data.copy_(weights[f"{layer_prefix}attn.output_proj.weight"])

        # RMSNorm
        ln1.weight.data.copy_(weights[f"{layer_prefix}ln1.weight"])
        ln2.weight.data.copy_(weights[f"{layer_prefix}ln2.weight"])

        # SwiGLU
        ffn.w1.weight.data.copy_(weights[f"{layer_prefix}ffn.w1.weight"])
        ffn.w2.weight.data.copy_(weights[f"{layer_prefix}ffn.w2.weight"])
        ffn.w3.weight.data.copy_(weights[f"{layer_prefix}ffn.w3.weight"])

    # final norm and lm head
    transformer_lm.ln_final.weight.data.copy_(weights["ln_final.weight"])
    transformer_lm.lm_head.linear.weight.data.copy_(weights["lm_head.weight"])

    return transformer_lm(in_indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """给定 RMSNorm 仿射变换的权重，返回对输入特征运行 RMSNorm 的输出。

    参数:
        d_model (int): RMSNorm 输入的维度。
        eps (float): 为数值稳定性添加到分母的值。
        weights (Float[Tensor, "d_model"]): RMSNorm 权重。
        in_features (Float[Tensor, "... d_model"]): 要运行 RMSNorm 的输入特征。
            可以有任意的领先维度。

    返回:
        Float[Tensor,"... d_model"]: 与 `in_features` 形状相同的张量，
        包含对 `in_features` 运行 RMSNorm 的输出。
    """
    from cs336_basics.model import RMSNorm
    norm=RMSNorm(d_model, eps)
    norm.weight.data.copy_(weights)
    return norm(in_features)



def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """给定输入张量，返回对每个元素应用 SiLU 的输出。

    参数:
        in_features (Float[Tensor, "..."]): 要运行 SiLU 的输入特征。形状任意。

    返回:
        Float[Tensor,"..."]: 与 `in_features` 形状相同的张量，
        包含应用 SiLU 的输出。
    """
    from cs336_basics.model import silu
    return silu(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    给定数据集（1D numpy 整数数组）和所需的 batch size 和 context length，
    从数据集中采样语言建模输入序列及其对应的标签。

    参数:
        dataset (np.array): 数据集中整数 token ID 的 1D numpy 数组。
        batch_size (int): 所需的 batch size。
        context_length (int): 每个采样示例的 context length。
        device (str): PyTorch 设备字符串（例如 'cpu' 或 'cuda:0'），
            表示放置采样输入序列和标签的设备。

    返回:
        shape 为 (batch_size, context_length) 的 torch.LongTensors 元组。
        第一个是采样的输入序列，第二个是对应的语言建模标签。
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    给定输入张量，返回在给定 `dim` 上 softmax 的输出。

    参数:
        in_features (Float[Tensor, "..."]): 要 softmax 的输入特征。形状任意。
        dim (int): 要应用 softmax 的 `in_features` 维度。

    返回:
        Float[Tensor, "..."]: 与 `in_features` 形状相同的张量，
        包含在指定维度上归一化的 softmax 输出。
    """
    from cs336_basics.training import softmax
    return softmax(in_features, dim)
    raise NotImplementedError


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """给定输入张量和目标张量，计算跨示例的平均交叉熵损失。

    参数:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] 是第 i 个样本第 j 类的
            未归一化 logit。
        targets (Int[Tensor, "batch_size"]): shape 为 (batch_size,) 的张量，
            包含正确类的索引。每个值必须在 0 和 `num_classes - 1` 之间。

    返回:
        Float[Tensor, ""]: 跨示例的平均交叉熵损失。
    """
    from cs336_basics.training import cross_entropy
    return cross_entropy(inputs, targets)
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """给定一组参数，将其组合梯度裁剪为最多 max_l2_norm 的 L2 范数。

    参数:
        parameters (Iterable[torch.nn.Parameter]): 可训练参数的集合。
        max_l2_norm (float): 包含最大 L2 范数的正值。

    梯度（parameter.grad）应原地修改。
    """
    from cs336_basics.training import gradient_clipping
    gradient_clipping(parameters, max_l2_norm)
    return


def get_adamw_cls() -> Any:
    """
    返回一个实现了 AdamW 的 torch.optim.Optimizer。
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    给定余弦学习率衰减计划（带线性 warmup）的参数和迭代数，
    返回在给定迭代下该计划的学习率。

    参数:
        it (int): 要获取学习率的迭代数。
        max_learning_rate (float): alpha_max，余弦学习率计划（带 warmup）的最大学习率。
        min_learning_rate (float): alpha_min，余弦学习率计划的最小/最终学习率。
        warmup_iters (int): T_w，线性 warmup 的迭代次数。
        cosine_cycle_iters (int): T_c，余弦退火迭代次数。

    返回:
        指定迭代下该计划的学习率。
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    给定模型、优化器和迭代数，将它们序列化到磁盘。

    参数:
        model (torch.nn.Module): 要序列化的模型状态。
        optimizer (torch.optim.Optimizer): 要序列化的优化器状态。
        iteration (int): 要序列化的值，表示已完成的训练迭代数。
        out (str | os.PathLike | BinaryIO | IO[bytes]): 序列化模型、优化器和迭代的目标路径或文件对象。
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    给定序列化的 checkpoint（路径或文件对象），将序列化的状态恢复到给定的模型和优化器。
    返回之前在 checkpoint 中序列化的迭代次数。

    参数:
        src (str | os.PathLike | BinaryIO | IO[bytes]): 序列化 checkpoint 的路径或文件对象。
        model (torch.nn.Module): 要恢复状态的模型。
        optimizer (torch.optim.Optimizer): 要恢复状态的优化器。

    返回:
        int: 之前序列化的迭代次数。
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """根据给定词表、BPE 合并列表和特殊 token，返回一个 BPE tokenizer。

    参数:
        vocab (dict[int, bytes]): tokenizer 词表。key 是 token ID，value 是该 token 对应的 bytes。
        merges (list[tuple[bytes, bytes]]): BPE 合并规则。每一项是两个 bytes token
            (<token1>, <token2>)，表示训练时曾把它们合并成 <token1><token2>。
            这个列表必须保留训练时的创建顺序，编码时也要按这个优先级应用。
        special_tokens (list[str] | None): 特殊 token 字符串列表。编码时这些字符串不能被拆开，
            必须始终作为单个 token 处理。

    返回:
        一个使用上述 vocab、merges 和 special_tokens 的 BPE tokenizer 实例。
    """
    from cs336_basics.tokenizer import Tokenizer
    return Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """根据输入语料路径训练 byte-level BPE tokenizer，并返回训练出的词表和合并规则。

    参数:
        input_path (str | os.PathLike): BPE tokenizer 训练语料的文本文件路径。
        vocab_size (int): 最终词表大小上限，包含 256 个初始 byte token、特殊 token，
            以及 BPE 合并产生的新 token。
        special_tokens (list[str]): 需要加入词表的特殊 token 字符串。训练时它们应该作为硬边界，
            防止两侧文本发生跨边界合并；它们本身不应该参与普通 pair 统计。

    返回:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                训练后的词表。key 是 token ID，value 是 token 对应的 bytes。
            merges:
                训练得到的 BPE 合并规则。每一项是两个 bytes token (<token1>, <token2>)，
                表示训练过程中把它们合并成了 <token1><token2>。
                merges 必须按创建顺序排列。
    """
    from cs336_basics.tokenizer import train_bpe
    return train_bpe(input_path, vocab_size, special_tokens, **kwargs)
