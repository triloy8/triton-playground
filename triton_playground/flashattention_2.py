"""FlashAttention-2 entrypoint living in the aggregator package.
"""

from __future__ import annotations

import os
import traceback

import torch
from einops import einsum

from flashattention_2 import FlashAttention2Torch, FlashAttention2Triton


def _softmax(x: torch.Tensor, dim: int):
    x_max = x.max(dim=dim, keepdim=True).values
    x_stable = x - x_max
    exp_x = torch.exp(x_stable)
    sum_exp_x = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp_x


def _vanilla_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    qk_score = einsum(Q, K, "batch_size ... nq d, batch_size ... nk d -> batch_size ... nq nk") / torch.sqrt(torch.tensor(Q.shape[-1]))
    masked_qk_score = qk_score.masked_fill(~mask, float('-inf'))
    softmax_masked_qk_score = _softmax(masked_qk_score, dim=-1)
    attn = einsum(softmax_masked_qk_score, V, "batch_size ... nq nk, batch_size ... nk d -> batch_size ... nq d")
    return attn


def _ensure_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _tinfo(x: torch.Tensor, name: str) -> str:
    return (
        f"{name}: shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
        f"contig={x.is_contiguous()} requires_grad={x.requires_grad}"
    )


def main() -> None:
    # Minimal, no-frills dry run scaffold
    bsz, heads, nq, nk, dk = 2, 2, 16, 16, 32
    dtype = torch.float32
    device = _ensure_device()
    g = torch.Generator(device=device if device != "cpu" else "cpu").manual_seed(0)

    Q = torch.randn((bsz, heads, nq, dk), generator=g, dtype=dtype, device=device if device != "cpu" else "cpu")
    K = torch.randn((bsz, heads, nk, dk), generator=g, dtype=dtype, device=device if device != "cpu" else "cpu")
    V = torch.randn((bsz, heads, nk, dk), generator=g, dtype=dtype, device=device if device != "cpu" else "cpu")

    # Full attention mask (no causal masking for now)
    mask = torch.ones((bsz, heads, nq, nk), dtype=torch.bool, device=Q.device)

    # Reference PyTorch attention
    ref = _vanilla_attention(Q, K, V, mask)

    # Try FlashAttention2Torch (may not be fully implemented yet)
    fa_out = None
    try:
        os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
        fa_out, fa_logsumexp = FlashAttention2Torch.apply(Q, K, V, False)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: WPS440
        print("[warn] FlashAttention2Torch.apply failed; using reference only.")
        print(f"error: {exc.__class__.__name__}: {exc}")
        print("traceback:")
        print(traceback.format_exc().rstrip())
        print("inputs:")
        print(_tinfo(Q, "Q"))
        print(_tinfo(K, "K"))
        print(_tinfo(V, "V"))

    fa_triton_out = None
    try:
        os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
        fa_triton_out, fa_triton_logsumexp = FlashAttention2Triton.apply(Q, K, V, False)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: WPS440
        print("[warn] FlashAttention2Triton.apply failed; using reference only.")
        print(f"error: {exc.__class__.__name__}: {exc}")
        print("traceback:")
        print(traceback.format_exc().rstrip())
        print("inputs:")
        print(_tinfo(Q, "Q"))
        print(_tinfo(K, "K"))
        print(_tinfo(V, "V"))

    print(
        "inputs:",
        f"bsz={bsz} heads={heads} nq={nq} nk={nk} dk={dk}",
        f"device={Q.device.type} dtype={str(dtype).split('.')[-1]}",
    )
    print(f"ref shape: {tuple(ref.shape)} mean={ref.float().mean().item():.4f} std={ref.float().std().item():.4f}")

    if fa_out is not None:
        diff = (fa_out.float() - ref.float()).abs().mean().item()
        print(
            f"fa shape:  {tuple(fa_out.shape)} mean={fa_out.float().mean().item():.4f} std={fa_out.float().std().item():.4f}",
        )
        print(f"mean abs diff (fa - ref): {diff:.6f}")

    if fa_triton_out is not None:
        diff = (fa_triton_out.float() - ref.float()).abs().mean().item()
        print(
            f"fa triton shape:  {tuple(fa_triton_out.shape)} mean={fa_triton_out.float().mean().item():.4f} std={fa_triton_out.float().std().item():.4f}",
        )
        print(f"mean abs diff (fa triton - ref): {diff:.6f}")


if __name__ == "__main__":
    main()
