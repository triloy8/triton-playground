import importlib.metadata

# Public API: re-export just your objects
from .weighted_sum_torch import WeightedSumFunc  # noqa: F401

__all__ = ["WeightedSumFunc"]