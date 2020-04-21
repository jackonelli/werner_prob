"""File IO utilities"""
from pathlib import Path
from typing import Union
import torch
from torch.optim import Optimizer
from src.models.interface import Classifier


def save_model(model: Classifier, loc: Union[str, Path], name: Union[str,
                                                                     Path]):
    """Save model to file

    Args:
        model (Classifier)
        loc (str/Path): Directory to save to, must exist.
        name (str/Path)
    """

    loc = Path(loc)
    if not loc.exists():
        raise IOError("Directory '{}' does not exist".format(loc))

    torch.save(model.state_dict(), loc / "{}.pth".format(name))


def load_model(model: Classifier, loc: Union[str, Path], name: Union[str,
                                                                     Path]):
    """Load model from file

    Args:
        model (Classifier): "Uninitialised" model
        loc (str/Path): Directory to save to, must exist.
        name (str/Path)
    """

    full_path = Path(loc) / name
    if not full_path.exists():
        raise IOError("Model file '{}' does not exist".format(full_path))

    if full_path.suffix != ".pth":
        raise IOError(
            "Model file '{}' must have '.pth' extension".format(full_path))

    model.load_state_dict(torch.load(full_path))
    return model


def save_optimizer(optimizer: Optimizer, loc: Union[str, Path],
                   name: Union[str, Path]):
    """Save optimizer to file

    Args:
        optimizer (Optimizer)
        loc (str/Path): Directory to save to, must exist.
        name (str/Path)
    """

    loc = Path(loc)
    if not loc.exists():
        raise IOError("Directory '{}' does not exist".format(loc))

    torch.save(optimizer.state_dict(), loc / "{}_optimizer.pth".format(name))
