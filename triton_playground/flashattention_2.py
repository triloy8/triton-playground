"""FlashAttention-2 entrypoint living in the aggregator package.
"""

from __future__ import annotations

import os
import traceback

import torch
from einops import einsum

from flashattention_2 import FlashAttention2Triton


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
    # Fixed config matching current setup
    bsz, heads, nq, nk, dk = 2, 2, 16, 16, 32
    dtype = torch.float32
    device = _ensure_device()
    is_causal = False

    # Deterministic local generator
    gen_device = device if device != "cpu" else "cpu"
    g = torch.Generator(device=gen_device).manual_seed(0)

    # Reference inputs (fp32, requires_grad)
    Q_ref = torch.randn((bsz, heads, nq, dk), generator=g, dtype=dtype, device=gen_device, requires_grad=True)
    K_ref = torch.randn((bsz, heads, nk, dk), generator=g, dtype=dtype, device=gen_device, requires_grad=True)
    V_ref = torch.randn((bsz, heads, nk, dk), generator=g, dtype=dtype, device=gen_device, requires_grad=True)

    # Causal mask
    mask = torch.ones((bsz, heads, nq, nk), dtype=torch.bool, device=Q_ref.device)
    if is_causal:
        mask = torch.tril(torch.ones(bsz, heads, nq, nk, dtype=torch.bool, device=Q_ref.device))

    # Reference forward
    O_ref = _vanilla_attention(Q_ref, K_ref, V_ref, mask)

    # Upstream gradient (deterministic)
    dO = torch.randn(O_ref.shape, generator=g, dtype=O_ref.dtype, device=O_ref.device)

    # Reference L = logsumexp(S) with same causal masking
    S_ref = einsum(Q_ref, K_ref, "b h nq d, b h nk d -> b h nq nk") / torch.sqrt(
        torch.tensor(dk, device=Q_ref.device, dtype=torch.float32)
    )
    if is_causal:
        S_ref = S_ref.masked_fill(~mask, float("-inf"))
    L_ref = torch.logsumexp(S_ref, dim=-1)

    # Closed-form forward + gradients via softmax for diagnostics
    scale = 1.0 / torch.sqrt(torch.tensor(dk, device=Q_ref.device, dtype=torch.float32))
    P_ref = torch.exp(S_ref - L_ref[..., None])
    # Forward (formula) output using P_ref
    O_form = einsum(P_ref, V_ref, "b h nq nk, b h nk d -> b h nq d")
    fwd_form_mae = (O_form.float() - O_ref.float()).abs().mean().item()
    dP_ref = einsum(dO, V_ref, "b h nq d, b h nk d -> b h nq nk")
    D_ref = (dO * O_ref).sum(dim=-1)
    dS_ref = P_ref * (dP_ref - D_ref[..., None]) * scale
    dV_form = einsum(P_ref, dO, "b h nq nk, b h nq d -> b h nk d")
    dK_form = einsum(dS_ref, Q_ref, "b h nq nk, b h nq d -> b h nk d")
    dQ_form = einsum(dS_ref, K_ref, "b h nq nk, b h nk d -> b h nq d")

    # Reference backward
    (O_ref * dO).sum().backward()
    dQ_ref, dK_ref, dV_ref = Q_ref.grad, K_ref.grad, V_ref.grad

    # Formula vs autograd MAEs
    form_dQ_mae = (dQ_form.float() - dQ_ref.float()).abs().mean().item()
    form_dK_mae = (dK_form.float() - dK_ref.float()).abs().mean().item()
    form_dV_mae = (dV_form.float() - dV_ref.float()).abs().mean().item()

    # Triton path (try/except)
    fwd_mae = dQ_mae = dK_mae = dV_mae = None
    l_mae = None
    try:
        Qt = Q_ref.detach().clone().requires_grad_(True)
        Kt = K_ref.detach().clone().requires_grad_(True)
        Vt = V_ref.detach().clone().requires_grad_(True)

        # Try to capture (O, L) if the Function returns both; else only O
        out = FlashAttention2Triton.apply(Qt, Kt, Vt, is_causal)  # type: ignore[arg-type]
        if isinstance(out, tuple) and len(out) == 2:
            O_tri, L_tri = out
            l_mae = (L_tri.float() - L_ref.float()).abs().mean().item()
        else:
            O_tri = out  # type: ignore[assignment]
        (O_tri * dO).sum().backward()

        dQ_tri, dK_tri, dV_tri = Qt.grad, Kt.grad, Vt.grad

        fwd_mae = (O_tri.float() - O_ref.float()).abs().mean().item()
        fwd_tform_mae = (O_tri.float() - O_form.float()).abs().mean().item()
        dQ_mae = (dQ_tri.float() - dQ_ref.float()).abs().mean().item() if dQ_tri is not None else float("nan")
        dK_mae = (dK_tri.float() - dK_ref.float()).abs().mean().item() if dK_tri is not None else float("nan")
        dV_mae = (dV_tri.float() - dV_ref.float()).abs().mean().item() if dV_tri is not None else float("nan")

        # Triton vs formula MAEs
        dQ_tform = (dQ_tri.float() - dQ_form.float()).abs().mean().item() if dQ_tri is not None else float("nan")
        dK_tform = (dK_tri.float() - dK_form.float()).abs().mean().item() if dK_tri is not None else float("nan")
        dV_tform = (dV_tri.float() - dV_form.float()).abs().mean().item() if dV_tri is not None else float("nan")
    except Exception as exc:  # noqa: WPS440
        print(f"[warn] Triton backward compare skipped: {exc.__class__.__name__}: {exc}")
        print("traceback:")
        print(traceback.format_exc().rstrip())

    # Reporting: single concise line
    dtype_name = str(dtype).split(".")[-1]
    shape_str = f"b={bsz} h={heads} nq={nq} nk={nk} d={dk}"
    if fwd_mae is None:
        print(
            f"backward-check: device={device} dtype={dtype_name} causal={is_causal} {shape_str} "
            "fwd_mae=- dQ_mae=- dK_mae=- dV_mae=- l_mae=-"
        )
    else:
        l_mae_str = "-" if l_mae is None else f"{l_mae:.6f}"
        print(
            "backward-check:",
            f"device={device} dtype={dtype_name} causal={is_causal} {shape_str}",
            f"fwd_mae={fwd_mae:.6f} dQ_mae={dQ_mae:.6f} dK_mae={dK_mae:.6f} dV_mae={dV_mae:.6f} l_mae={l_mae_str}",
        )
        # Forward diagnostics
        fwd_tform_str = "-" if 'fwd_tform_mae' not in locals() else f"{fwd_tform_mae:.6f}"
        print(
            "forward-formula:",
            f"Of_mae={fwd_form_mae:.6f}",
        )
        print(
            "triton-forward-formula:",
            f"Otf_mae={fwd_tform_str}",
        )
        # Extra diagnostics lines (formula comparisons)
        print(
            "formula-autograd:",
            f"dQf_mae={form_dQ_mae:.6f} dKf_mae={form_dK_mae:.6f} dVf_mae={form_dV_mae:.6f}",
        )
        print(
            "triton-formula:",
            f"dQt_mae={dQ_tform:.6f} dKt_mae={dK_tform:.6f} dVt_mae={dV_tform:.6f}",
        )


if __name__ == "__main__":
    main()
