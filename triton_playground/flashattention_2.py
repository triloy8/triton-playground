"""FlashAttention-2 entrypoint living in the aggregator package.
"""

from __future__ import annotations

import torch
from einops import einsum

from flashattention_2 import FlashAttention2Torch


def _softmax(x: torch.Tensor, dim: int):
    x_max = x.max(dim=dim, keepdim=True).values
    x_stable = x - x_max
    exp_x = torch.exp(x_stable)
    sum_exp_x = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp_x


def _vanilla_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    qk_score = einsum(Q, K, "batch_size ... n d_k, batch_size ... m d_k -> batch_size ... n m") / torch.sqrt(torch.tensor(Q.shape[-1]))
    masked_qk_score = qk_score.masked_fill(~mask, float('-inf'))
    softmax_masked_qk_score = _softmax(masked_qk_score, dim=-1)
    attn = einsum(softmax_masked_qk_score, V, "batch_size ... n m, batch_size ... m d_k -> batch_size ... n d_k")
    return attn


def main() -> None:
    pass


if __name__ == "__main__":
    main()
