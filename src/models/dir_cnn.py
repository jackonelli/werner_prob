"""CNN direction estimation"""
import torch.nn as nn
import torch.nn.functional as F
from src.models.interface import Classifier


class Net(Classifier):
    """CNN direction estimation"""
    def __init__(self):
        super(Net, self).__init__()
        self.name = "CNN direction"
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def info(self):
        """Model info to string"""
        return self.name

    def forward(self, x):
        """Propagate input"""
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
