import torch
import triton
from einops import einsum
import math
from einops import rearrange

from .flashattention_2_tl import flashattention_2_fwd, flashattention_2_bwd_dkv, flashattention_2_bwd_dq

class FlashAttention2Torch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Bq = 16
        Bk = 16

        Tq =  math.ceil(Q.shape[-2] / Bq)
        Tk =  math.ceil(K.shape[-2] / Bk)

        O = torch.zeros(*Q.shape, device=Q.device, dtype=Q.dtype)
        L = torch.zeros(*Q.shape[:-1], device=Q.device, dtype=Q.dtype)
        M = torch.full(Q.shape[:-1], float('-inf'), device=Q.device, dtype=Q.dtype)

        Q_list = torch.chunk(Q, Tq, dim=-2)
        K_list = torch.chunk(K, Tk, dim=-2)
        V_list = torch.chunk(V, Tk, dim=-2)
        O_list = torch.chunk(O, Tq, dim=-2)
        L_list = torch.chunk(L, Tq, dim=-1)
        M_list = torch.chunk(M, Tq, dim=-1)

        for i in range(0, Tq):
            l_i = torch.zeros(*L_list[0].shape, device=L_list[0].device, dtype=L_list[0].dtype)
            for j in range(0, Tk):
                M_i_jm1 = M_list[i].clone()
                O_i_jm1 = O_list[i].clone()
                l_i_jm1 = l_i.clone()
                
                # compute score on given tiles
                S = einsum(Q_list[i], K_list[j], 
                           "batch_size ... n d, batch_size ... m d -> batch_size ... n m") / torch.sqrt(torch.tensor(Q_list[i].shape[-1]))
                if is_causal:
                    mask = torch.tril(torch.ones(*S.shape, dtype=torch.bool, device=Q.device))
                    S = S .masked_fill(~mask, float('-inf'))
                # compute row score maxes
                S_rowmax, _ = torch.max(S, dim=-1)
                # update max running value
                torch.maximum(M_list[i], S_rowmax, out=M_list[i])
                # new numerator partial softmax
                P = torch.exp(S - M_list[i][..., None])
                # update w/ denominator partial softmax
                l_i = einsum(torch.exp(M_i_jm1 - M_list[i]), l_i_jm1,
                             "batch_size ... n, batch_size ... n -> batch_size ... n") + torch.sum(P, dim=-1)
                # update output w/ numerator and update max
                O_i = O_list[i]
                upd_o = einsum(torch.exp(M_i_jm1 - M_list[i]), O_i_jm1,
                               "batch_size ... n, batch_size ... n d -> batch_size ... n d")
                partial_o = einsum(P, V_list[j],
                                   "batch_size ... n m, batch_size ... m d -> batch_size ... n d")
                O_i.copy_(upd_o+partial_o)
            # update final output w/ softmax denominator
            O_i = O_list[i]
            final_o_i = einsum(1/l_i, O_i,
                               "batch_size ... n, batch_size ... n d -> batch_size ... n d")
            O_i.copy_(final_o_i)
            # update log sum exp for rows
            L_i = L_list[i]
            final_l_i = einsum(M_list[i], torch.log(l_i),
                               "batch_size ... n, batch_size ... n -> batch_size ... n")
            L_i.copy_(final_l_i)

        O = torch.cat(O_list, dim=-2)
        L = torch.cat(L_list, dim=-1)

        return O, L

    @staticmethod
    def backward(ctx):
        raise NotImplementedError


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Bq = 16
        Bk = 16
        Tq =  math.ceil(Q.shape[-2] / Bq)
        Tk =  math.ceil(K.shape[-2] / Bk)

        B, H, Nq, Nk, D = Q.shape[0], Q.shape[1], Q.shape[2], K.shape[2], Q.shape[3]

        Q = rearrange(Q, 'b h nq d -> (b h) nq d').contiguous()
        K = rearrange(K, 'b h nk d -> (b h) d nk').contiguous()
        V = rearrange(V, 'b h nk d -> (b h) nk d').contiguous()

        O = torch.zeros(*Q.shape, device=Q.device, dtype=Q.dtype)
        L = torch.zeros(*Q.shape[:-1], device=Q.device, dtype=Q.dtype)

        N_QUERIES = Q.shape[-2]
        N_KEYS = K.shape[-1]
        scale = 1 / math.sqrt(Q.shape[-1])
        D = Q.shape[-1]
        Q_TILE_SIZE = triton.next_power_of_2(N_QUERIES) // Tq
        K_TILE_SIZE = triton.next_power_of_2(N_KEYS) // Tk

        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.d = D
        ctx.B = B
        ctx.H = H
        ctx.N_QUERIES = N_QUERIES
        ctx.N_KEYS = N_KEYS
        ctx.Q_TILE_SIZE = Q_TILE_SIZE
        ctx.K_TILE_SIZE = K_TILE_SIZE

        grid = (triton.cdiv(N_QUERIES, Q_TILE_SIZE), ctx.B*ctx.H)
        flashattention_2_fwd[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(-2), Q.stride(-1),
            K.stride(0), K.stride(-1), K.stride(-2),
            V.stride(0), V.stride(-2), V.stride(-1),
            O.stride(0), O.stride(-2), O.stride(-1),
            L.stride(0), L.stride(-1),
            N_QUERIES, N_KEYS,
            scale,
            D,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal,
        )

        Q = rearrange(Q, '(b h) nq d -> b h nq d', b=ctx.B, h=ctx.H).contiguous()
        K = rearrange(K, '(b h) d nk -> b h nk d', b=ctx.B, h=ctx.H).contiguous()
        V = rearrange(V, '(b h) nk d -> b h nk d', b=ctx.B, h=ctx.H).contiguous()
        L = rearrange(L, '(b h) nq -> b h nq', b=ctx.B, h=ctx.H).contiguous()
        O = rearrange(O, '(b h) nq d -> b h nq d', b=ctx.B, h=ctx.H).contiguous()

        ctx.save_for_backward(Q, K, V, O, L)

        return O, L

    @staticmethod
    def backward(ctx, dO, dL=None):  # dL is unused
        Q, K, V, O, L = ctx.saved_tensors

        Q = rearrange(Q, 'b h nq d -> (b h) nq d').contiguous()
        K = rearrange(K, 'b h nk d -> (b h) nk d').contiguous()
        V = rearrange(V, 'b h nk d -> (b h) nk d').contiguous()
        O = rearrange(O, 'b h nq d -> (b h) nq d').contiguous()
        L = rearrange(L, 'b h nq -> (b h) nq').contiguous()
        dO = rearrange(dO, 'b h nq d -> (b h) nq d').contiguous()

        D = torch.sum(dO * O, dim=-1)

        dQ = torch.zeros(*Q.shape, device=Q.device, dtype=Q.dtype)
        dK = torch.zeros(*K.shape, device=K.device, dtype=K.dtype)
        dV = torch.zeros(*V.shape, device=V.device, dtype=V.dtype)

        grid_dkv = (triton.cdiv(ctx.N_KEYS, ctx.K_TILE_SIZE), ctx.B*ctx.H)
        flashattention_2_bwd_dkv[grid_dkv](
            dK, dV,
            dO,
            Q, K, V,
            L,
            D,
            dK.stride(0), dK.stride(-2), dK.stride(-1),
            dV.stride(0), dV.stride(-2), dV.stride(-1),
            dO.stride(0), dO.stride(-2), dO.stride(-1),
            Q.stride(0), Q.stride(-2), Q.stride(-1),
            K.stride(0), K.stride(-2), K.stride(-1),
            V.stride(0), V.stride(-2), V.stride(-1),
            L.stride(0), L.stride(-1),
            D.stride(0), D.stride(-1),
            ctx.N_QUERIES, ctx.N_KEYS,
            ctx.scale,
            ctx.d,
            ctx.Q_TILE_SIZE,
            ctx.K_TILE_SIZE,
            ctx.is_causal,
        )

        grid_dq = (triton.cdiv(ctx.N_QUERIES, ctx.Q_TILE_SIZE), ctx.B*ctx.H)
        flashattention_2_bwd_dq[grid_dq](
            dQ,
            dO,
            Q, K, V,
            L,
            D,
            dQ.stride(0), dQ.stride(-2), dQ.stride(-1),
            dO.stride(0), dO.stride(-2), dO.stride(-1),
            Q.stride(0), Q.stride(-2), Q.stride(-1),
            K.stride(0), K.stride(-2), K.stride(-1),
            V.stride(0), V.stride(-2), V.stride(-1),
            L.stride(0), L.stride(-1),
            D.stride(0), D.stride(-1),
            ctx.N_QUERIES, ctx.N_KEYS,
            ctx.scale,
            ctx.d,
            ctx.Q_TILE_SIZE,
            ctx.K_TILE_SIZE,
            ctx.is_causal,
        )

        dQ = rearrange(dQ, '(b h) nq d -> b h nq d', b=ctx.B, h=ctx.H).contiguous()
        dK = rearrange(dK, '(b h) nk d -> b h nk d', b=ctx.B, h=ctx.H).contiguous()
        dV = rearrange(dV, '(b h) nk d -> b h nk d', b=ctx.B, h=ctx.H).contiguous()

        return dQ, dK, dV, None
