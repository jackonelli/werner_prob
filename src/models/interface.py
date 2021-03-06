"""Classifier interface"""
from abc import ABC, abstractmethod
import torch


class Classifier(torch.nn.Module, ABC):
    """Classifier interface"""
    @abstractmethod
    def info(self):
        """Model info to string"""
        pass

    @abstractmethod
    def forward(self, input_):
        """Make prediction

        Compute class probabilities
        """
        pass
