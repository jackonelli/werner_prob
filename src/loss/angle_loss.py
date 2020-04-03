"""Loss module"""
import torch
from src.loss.interface import Loss as Interface


class Loss(Interface):
    """Angle loss"""
    def __init__(self, ref_vecs):
        self.name = "Angle"
        self._ref_vecs = ref_vecs

    def info(self):
        """Loss info to string"""
        return self.name

    def compute(self, estimate: torch.Tensor, target: torch.Tensor):
        """Compute loss"""
        angles_to_refs = angle(estimate, self._ref_vecs)


def cos_of_angle_to_refs(feature_vec: torch.Tensor,
                         ref_vecs: torch.Tensor) -> torch.Tensor:
    """Calculate the cosine of the angle between two vectors
    D - dimensionality of feature space
    K - number of reference vectors

    cos(alpha) = a dot b / (|a| |b|)

    Args:
        feature_vec (D, ): Single vector in feature space
        ref_vecs (K, D): Single vector in feature space
    """
    inner_products = (feature_vec * ref_vecs).sum(-1)
    feature_norm = feature_vec.norm()
    # Ordinary norm() cannot choose dimension?
    ref_norm = ref_vecs.pow(2).sum(-1).sqrt()
    return inner_products / (feature_norm * ref_norm)


def angle(a_vec: torch.Tensor, b_vec: torch.Tensor) -> torch.Tensor:
    """Calculate angle between two vectors"""
    return torch.acos(cos_of_angle_to_refs(a_vec, b_vec))
