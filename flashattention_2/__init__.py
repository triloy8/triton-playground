import importlib.metadata

# Public API: re-export just your objects
from .flashattention_2_torch import FlashAttention2Torch  # noqa: F401

__all__ = ["FlashAttention2Torch"]