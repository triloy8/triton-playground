import torch

class FlashAttention2Torch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        pass

    @staticmethod
    def backward(ctx):
        raise NotImplementedError