"""MNIST"""
import torch.utils.data as torch_data
import torchvision


def trainloader(batch_size_train: int) -> torch_data.DataLoader:
    """Generate trainloader"""
    return torch_data.DataLoader(torchvision.datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
        ])),
                                 batch_size=batch_size_train,
                                 shuffle=True)


def testloader(batch_size_test: int) -> torch_data.DataLoader:
    """Generate testloader"""
    return torch_data.DataLoader(torchvision.datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
        ])),
                                 batch_size=batch_size_test,
                                 shuffle=True)
