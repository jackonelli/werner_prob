"""Cross entropy loss"""
import torch
import torch.nn.functional as F
from src.loss.interface import Loss as Interface


class Loss(Interface):
    """Cross entropy loss"""
    def __init__(self):
        self._func = F.nll_loss

    def compute(self, estimate: torch.Tensor, target: torch.Tensor):
        """Compute loss"""
        return self._func(estimate, target)
