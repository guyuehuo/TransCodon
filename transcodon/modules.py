"""Modules used on the ProteinBERT model.

This code has been modified from the original implementation
by Facebook Research, describing its ESM-1b paper."""


import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multihead_attention import MultiheadAttention  # noqa


def gelu(x):
    """Implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    class ESM1bLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)


except ImportError:
    from torch.nn import LayerNorm as ESM1bLayerNorm


class TransformerLayer(nn.Module):
    """Transformer layer block."""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        attention_dropout=0.,
        add_bias_kv=True,
        use_esm1b_layer_norm=False,
        rope_embedding=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout
        self.rope_embedding = rope_embedding
        self._init_submodules(add_bias_kv, use_esm1b_layer_norm)

    def _init_submodules(self, add_bias_kv, use_esm1b_layer_norm):
        BertLayerNorm = ESM1bLayerNorm if use_esm1b_layer_norm else nn.LayerNorm

        self.self_attn = MultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            dropout=self.attention_dropout,
            rope_embedding=self.rope_embedding,
        )
        self.self_attn_layer_norm = BertLayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = BertLayerNorm(self.embed_dim)

    def forward(
        self, x, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x, attn


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super().__init__(num_embeddings_, embedding_dim, padding_idx)
        self.max_positions = num_embeddings

    def forward(self, input: torch.Tensor):
        """Input is expected to be of size [bsz x seqlen]."""
        if input.size(1) > self.max_positions:
            raise ValueError(
                f"Sequence length {input.size(1)} above maximum "
                f" sequence length of {self.max_positions}"
            )
        mask = input.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, weight):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = ESM1bLayerNorm(embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x
    
class RNA_2D_RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = ESM1bLayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, output_dim)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = self.out_proj(x)  + self.bias
        return x

class RelativePositionEncoder(nn.Module):
    """相对位置编码模块"""
    def __init__(self, max_rel_pos=20):
        super().__init__()
        self.max_rel_pos = max_rel_pos
        # 相对位置嵌入层：覆盖[-max_rel_pos, max_rel_pos]范围
        self.rel_pos_emb = nn.Embedding(max_rel_pos + 1, 1)
        
    def forward(self, seq_len, device):
        """生成相对位置偏置矩阵(优化内存版本)"""
        # 预先生成位置索引 (L)
        pos = torch.arange(seq_len, device=device)
        
        # 动态计算距离矩阵 (L, L)
        rel_pos = pos[:, None] - pos[None, :]            # (L, L)
        rel_pos = torch.abs(rel_pos)                     # 取绝对值
        rel_pos = torch.clamp(rel_pos, 0, self.max_rel_pos)  # 限制在[0, max_rel_pos]
        
        # 直接查询嵌入表 (避免中间矩阵存储)
        return self.rel_pos_emb(rel_pos).squeeze(-1)     # (L, L)

class RNA_2D_Structure_Connector(nn.Module):
    def __init__(self, embed_dim, max_rel_pos=20):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_rel_pos = max_rel_pos
        # 共享投影矩阵 (减少参数)
        self.shared_proj = nn.Linear(embed_dim, embed_dim)
        # 相对位置编码器
        self.rel_pos_encoder = RelativePositionEncoder(max_rel_pos)
        #self.null_bias_layer = nn.Linear(embed_dim, 1)  # 通过输入嵌入生成null偏置
        # 简化无连接预测头
        self.null_bias_layer = nn.Linear(embed_dim, 1)

    def forward(self, codon_embeddings):
        B, L, E = codon_embeddings.shape
        # 共享投影计算 (减少一次矩阵乘法)
        proj = self.shared_proj(codon_embeddings)        # (B, L, E)
        # 分解计算注意力分数 (B, L, L)
        position_scores = torch.bmm(proj, proj.transpose(1, 2))  # (B, L, L)

        #print("position_scores",position_scores.shape)
        position_scores += self.rel_pos_encoder(L, codon_embeddings.device)
        if False:
            print("before position_scores",position_scores.shape)
            # 添加相对位置偏置
            rel_pos_bias = self.rel_pos_encoder(L, codon_embeddings.device)  # (E, L, L)
            print("rel_pos_bias",rel_pos_bias.shape)
            print("queries",queries.shape)
            print("queries.unsqueeze(2)",queries.unsqueeze(2).shape)
            #rel_pos_bias = rel_pos_bias.permute(1, 2, 0)  # 变为 [1200, 1200, 768]
            
            position_scores += (queries.unsqueeze(2) * rel_pos_bias).sum(-1)  # (B, L, L)
            print("after position_scores",position_scores.shape)

        # diagonal_indices = torch.eye(L, dtype=torch.bool).to(codon_embeddings.device)
        # position_scores[:,diagonal_indices] = 0

        # 原位掩码操作 (节省内存)
        mask = torch.eye(L, dtype=torch.bool, device=position_scores.device)
        position_scores.masked_fill_(mask, 0.0)          # In-place操作
        #         # 将对角线位置的值设为 0
        #         struct_label_output[:,diagonal_indices] = 0
        null_scores = self.null_bias_layer(codon_embeddings)  # (B, L, 1)
        struct_logits = torch.cat([null_scores, position_scores], dim=-1)  # (B, L, L+1)
        return struct_logits

class ESM1LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12, affine=True):
        """Construct a layernorm layer in the TF style (eps inside the sqrt)."""
        super().__init__()
        self.hidden_size = (hidden_size,) if isinstance(hidden_size, int) else tuple(hidden_size)
        self.eps = eps
        self.affine = bool(affine)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.weight, self.bias = None, None

    def forward(self, x):
        dims = tuple(-(i + 1) for i in range(len(self.hidden_size)))
        means = x.mean(dims, keepdim=True)
        x_zeromean = x - means
        variances = x_zeromean.pow(2).mean(dims, keepdim=True)
        x = x_zeromean / torch.sqrt(variances + self.eps)
        if self.affine:
            x = (self.weight * x) + self.bias
        return x




try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    class ESM1bLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)


except ImportError:
    from torch.nn import LayerNorm as ESM1bLayerNorm

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, padding_idx, learned=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.weights = None

    def forward(self, x):
        bsz, seq_len = x.shape
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = self.get_embedding(max_pos)
        self.weights = self.weights.type_as(self._float_tensor)

        positions = self.make_positions(x)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def make_positions(self, x):
        mask = x.ne(self.padding_idx)
        range_buf = torch.arange(x.size(1), device=x.device).expand_as(x) + self.padding_idx + 1
        positions = range_buf.expand_as(x)
        return positions * mask.long() + self.padding_idx * (1 - mask.long())

    def get_embedding(self, num_embeddings):
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if self.embed_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if self.padding_idx is not None:
            emb[self.padding_idx, :] = 0
        return emb

class ContactPredictionHead(nn.Module):
    """Performs symmetrization, apc, and computes a logistic regression on the output features"""

    def __init__(
        self,
        in_features: int,
        prepend_bos: bool,
        append_eos: bool,
        bias=True,
        eos_idx: Optional[int] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        if append_eos and eos_idx is None:
            raise ValueError("Using an alphabet with eos token, but no eos token was passed in.")
        self.eos_idx = eos_idx
        self.regression = nn.Linear(in_features, 1, bias)
        self.activation = nn.Sigmoid()

    def forward(self, tokens, attentions):
        # remove eos token attentions
        if self.append_eos:
            eos_mask = tokens.ne(self.eos_idx).to(attentions)
            eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
            attentions = attentions * eos_mask[:, None, None, :, :]
            attentions = attentions[..., :-1, :-1]
        # remove cls token attentions
        if self.prepend_bos:
            attentions = attentions[..., 1:, 1:]
        batch_size, layers, heads, seqlen, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)

        # features: B x C x T x T
        attentions = attentions.to(
            next(self.parameters())
        )  # attentions always float32, may need to convert to float16
        attentions = apc(symmetrize(attentions))
        attentions = attentions.permute(0, 2, 3, 1)
        return self.activation(self.regression(attentions).squeeze(3))

