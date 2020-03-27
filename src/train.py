"""Train module"""
from typing import Dict, NamedTuple
import torch
import torch.utils.data as torch_data
from src.models.interface import Classifier
from src.loss.interface import Loss


def train(model: Classifier, dataloaders: Dict[str, torch_data.DataLoader],
          loss_function: Loss, optimizer, settings: "TrainSettings"):
    """Train model"""
    print("Training")
    for epoch in range(settings.num_epochs):
        train_loss = _train_epoch(model,
                                  dataloaders["train"],
                                  loss_function,
                                  optimizer,
                                  log_interval=settings.log_interval)
        validation_loss = None
        if "validate" in dataloaders:
            with torch.no_grad():
                validation_loss = _validate_epoch(
                    model,
                    dataloaders["validate"],
                    loss_function,
                )
        _print_epoch(epoch, settings.num_epochs, train_loss, validation_loss)


def _train_epoch(model: Classifier, dataloader: torch_data.DataLoader,
                 loss_function: Loss, optimizer, log_interval: int):
    store_loss = list()
    for batch_ind, batch in enumerate(dataloader):
        optimizer.zero_grad()
        input_, target = batch
        output = model(input_)
        loss = loss_function.compute(output, target)
        loss.backward()
        optimizer.step()
        _print_batch(batch_ind, log_interval, len(dataloader), loss.item())
        store_loss.append(loss.item())
    return torch.Tensor(store_loss)


def _validate_epoch(model, dataloader, loss_function):
    store_loss = list()
    for _, batch in enumerate(dataloader):
        input_, target = batch
        output = model(input_)
        store_loss.append(loss_function.compute(output, target).item())
    return torch.Tensor(store_loss)


def _print_epoch(current_epoch, total_epochs, train_loss, validation_loss):
    """Temp logger function"""
    train_loss_agg = train_loss.mean().item()
    val_loss_agg = validation_loss.mean().item() if validation_loss else "-"
    print("Epoch: {}/{}\tTrain loss: {}, Val. loss: {}".format(
        current_epoch, total_epochs, train_loss_agg, val_loss_agg))


def _print_batch(batch_ind: int, log_interval: int, num_batches: int,
                 loss: float):
    """Temp logger function"""
    if batch_ind % log_interval == 0 and batch_ind > 0:
        print("\tBatch: [{}/{}]\tLoss: {:.6f}".format(batch_ind, num_batches,
                                                      loss))


class TrainSettings(NamedTuple):
    """Train settings"""
    log_interval: int
    num_epochs: int
