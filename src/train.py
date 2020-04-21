"""Train module"""
from typing import Dict, NamedTuple, Optional
import logging
import torch
import torch.utils.data as torch_data
from src.models.interface import Classifier
from src.loss.interface import Loss

TRAIN_KEY = "train"
VAL_KEY = "validation"
LOGGER = logging.getLogger(__name__)


def train(model: Classifier, dataloaders: Dict[str, torch_data.DataLoader],
          loss_function: Loss, optimizer, settings: "TrainSettings"):
    """Train model"""
    LOGGER.info("Training - Model: {}, loss: {}, optimizer: {}".format(
        model.info(), loss_function.info(), optimizer))
    model = model.to(settings.device)
    for epoch in range(1, settings.num_epochs + 1):
        train_loss = _train_epoch(model,
                                  dataloaders[TRAIN_KEY],
                                  loss_function,
                                  optimizer,
                                  device=settings.device,
                                  log_interval=settings.log_interval)
        validation_loss = None
        if VAL_KEY in dataloaders:
            with torch.no_grad():
                validation_loss = _validate_epoch(model,
                                                  dataloaders[VAL_KEY],
                                                  loss_function,
                                                  device=settings.device)
        LOGGER.info(
            _epoch_summary(epoch, settings.num_epochs, train_loss,
                           validation_loss))
    return model


def _train_epoch(model: Classifier, dataloader: torch_data.DataLoader,
                 loss_function: Loss, optimizer, device: torch.device,
                 log_interval: int):
    model.train()
    store_loss = list()
    for batch_ind, batch in enumerate(dataloader):
        optimizer.zero_grad()
        input_, target = batch
        input_, target = input_.to(device), target.to(device)
        output = model(input_)
        loss = loss_function.compute(output, target)
        loss.backward()
        optimizer.step()
        batch_summary = _batch_summary(batch_ind, log_interval,
                                       len(dataloader), loss.item())
        if batch_summary is not None:
            LOGGER.info(batch_summary)
        store_loss.append(loss.item())
    return torch.Tensor(store_loss)


def _validate_epoch(model: Classifier, dataloader: torch_data.DataLoader,
                    loss_function: Loss, device: torch.device):
    model.eval()
    store_loss = list()
    for _, batch in enumerate(dataloader):
        input_, target = batch
        input_, target = input_.to(device), target.to(device)
        output = model(input_)
        store_loss.append(loss_function.compute(output, target).item())
    return torch.Tensor(store_loss)


def _epoch_summary(current_epoch, total_epochs, train_loss, validation_loss):
    """Temp logger function"""
    train_loss_agg = train_loss.mean().item()
    val_loss_agg = validation_loss.mean().item(
    ) if validation_loss is not None else "-"
    return "Epoch: {}/{}\tTrain loss: {:3f}, Val. loss: {}".format(
        current_epoch, total_epochs, train_loss_agg, val_loss_agg)


def _batch_summary(batch_ind: int, log_interval: int, num_batches: int,
                   loss: float) -> Optional[str]:
    """Temp logger function"""
    if batch_ind % log_interval == 0 and batch_ind > 0:
        return "\tBatch: [{}/{}]\tLoss: {:.3f}".format(batch_ind, num_batches,
                                                       loss)
    else:
        return None


class TrainSettings(NamedTuple):
    """Train settings"""
    log_interval: int
    num_epochs: int
    device: torch.device
