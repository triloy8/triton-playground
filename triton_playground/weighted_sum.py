"""Weighted-sum demo entrypoint living in the aggregator package.

This script prefers the Triton implementation when CUDA is available,
and falls back to a pure PyTorch implementation otherwise.
"""

from __future__ import annotations

import time

import torch


def _ensure_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _torch_impl(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # Shape: (rows, dims) • (dims,) -> (rows,)
    return torch.einsum("rd,d->r", x, w)


def main() -> None:
    # Defaults: edit to taste
    rows, dims = 1024, 512
    dtype = torch.float32
    device = _ensure_device()
    g = torch.Generator(device=device).manual_seed(0)

    x = torch.randn((rows, dims), generator=g, dtype=dtype, device=device if device != "cpu" else "cpu")
    w = torch.randn((dims,), generator=g, dtype=dtype, device=device if device != "cpu" else "cpu")

    # Prefer Triton implementation if available
    triton_func = None
    if device.startswith("cuda"):
        try:
            from weighted_sum import WeightedSumFunc  # type: ignore

            triton_func = WeightedSumFunc
        except Exception as exc:  # noqa: WPS440
            print(f"[warn] Triton version unavailable, using PyTorch. ({exc})")

    if triton_func is not None and x.device.type != "cuda":
        x = x.to("cuda")
        w = w.to("cuda")

    # Warmup
    for _ in range(2):
        if triton_func is not None:
            _ = triton_func.apply(x, w)
        else:
            _ = _torch_impl(x, w)

    start = time.perf_counter()
    if triton_func is not None:
        y = triton_func.apply(x, w)
        torch.cuda.synchronize()
    else:
        y = _torch_impl(x, w)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"device={x.device.type} rows={rows} dims={dims} dtype={str(dtype).split('.')[-1]} time_ms={elapsed_ms:.3f}")
    print(f"output shape: {tuple(y.shape)} mean={y.float().mean().item():.4f} std={y.float().std().item():.4f}")


if __name__ == "__main__":
    main()
