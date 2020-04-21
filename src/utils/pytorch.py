"""Pytorch GPU utilities"""
import logging
from pathlib import Path
from typing import Union
import torch
from torch.optim import Optimizer
from src.models.interface import Classifier

LOGGER = logging.getLogger(__name__)


def cuda_settings(use_gpu):
    """Pytorch settings"""
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    return device


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
        err = "Directory '{}' does not exist".format(loc)
        LOGGER.error(err)
        raise IOError(err)

    model_path = loc / "{}.pth".format(name)
    torch.save(model.state_dict(), model_path)
    LOGGER.info("Model saved to '{}'.".format(model_path))


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
        err = "Model file '{}' does not exist".format(full_path)
        LOGGER.error(err)
        raise IOError(err)

    if full_path.suffix != ".pth":
        err = "Model file '{}' must have '.pth' extension".format(full_path)
        LOGGER.error(err)
        raise IOError(err)

    model.load_state_dict(torch.load(full_path))
    LOGGER.info("Model loaded from '{}'.".format(full_path))
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
        err = "Directory '{}' does not exist".format(loc)
        LOGGER.error(err)
        raise IOError(err)
    optimizer_path = loc / "{}_optimizer.pth".format(name)
    torch.save(optimizer.state_dict(), optimizer_path)
    LOGGER.info("Optimizer saved to '{}'.".format(optimizer_path))
