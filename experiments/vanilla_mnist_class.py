"""Training on MNIST

Inspo: https://nextjournal.com/gkoehler/pytorch-mnist
"""
import torch
import torch.optim as optim
from src.train import train, TrainSettings
import src.models.cnn as cnn
from src.loss.cross_entropy import Loss as XeLoss
from src.dataloaders import mnist


def main():
    """Main entry point for script"""
    num_epochs = 3
    batch_size_train = 64
    # batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    # log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    model = cnn.Net()
    loss_function = XeLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=momentum)

    train_settings = TrainSettings(log_interval=1000, num_epochs=num_epochs)
    train(model, {"train": mnist.trainloader(batch_size_train)}, loss_function,
          optimizer, train_settings)


if __name__ == "__main__":
    main()
