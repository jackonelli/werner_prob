"""Loss interface"""
from abc import ABC, abstractmethod
import torch


class Loss(ABC):
    """Loss interface"""
    @abstractmethod
    def compute(self, estimate: torch.Tensor, target: torch.Tensor):
        """Compute loss"""
        pass
