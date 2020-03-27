"""Loss module"""
import torch


def cos_of_angle_to_refs(feature_vec: torch.Tensor,
                         ref_vecs: torch.Tensor) -> torch.Tensor:
    """Calculate angle between two vectors
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


def quick_test():
    x_vec = torch.Tensor([1.0, 0.0])
    y_vec = torch.Tensor([[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0]])
    aba = (x_vec * y_vec).sum(-1)
    y_norm = y_vec.pow(2).sum(-1).sqrt()
    print("cos_ref", cos_of_angle_to_refs(x_vec, y_vec))


if __name__ == "__main__":
    quick_test()
