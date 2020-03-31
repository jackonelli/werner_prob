"""Cross entropy loss"""
import torch
import torch.nn.functional as F
from src.loss.interface import Loss as Interface


class Loss(Interface):
    """Cross entropy loss"""
    def __init__(self):
        pass

    @staticmethod
    def compute(estimate: torch.Tensor, target: torch.Tensor):
        """Compute loss"""
        return F.nll_loss(estimate, target)
