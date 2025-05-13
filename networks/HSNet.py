import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
import logging
import numpy as np


class BaseNet(nn.Module):
    """Base class for all neural networks."""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rep_dim = None  # representation dimensionality, i.e. dim of the last layer

    def forward(self, *input):
        """
        Forward pass logic
        :return: Network output
        """
        raise NotImplementedError

    def summary(self):
        """Network summary."""
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)

class HSNet(BaseNet):

    def __init__(self):

        super().__init__()

        # Input parameters
        self.input_channels = 31
        self.patch_size = 32
        self.alpha = 0.1 # Leaky ReLU slope, default is 0.01

        # Representation dimensionality
        self.rep_dim = 128  # Output dimension of the last fully connected layer

        # Convolutional layers
        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=3, stride=1, padding=1)  # (31 -> 64 channels)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)             # (64 -> 128 channels)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)            # (128 -> 256 channels)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.bn2 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.bn3 = nn.BatchNorm2d(256, eps=1e-04, affine=False)

        # Fully connected layers
        flattened_size = 256 * (self.patch_size // 2 // 2 // 2) ** 2                            # After 3x2 max-pooling layers
        self.fc1 = nn.Linear(flattened_size, self.rep_dim, bias=False)  # Fully connected layer
        
        # Activation function
        self.leaky_relu = nn.LeakyReLU(self.alpha)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):

        # Convolutional layers with Leaky ReLU, batch normalization, and pooling
        x = self.conv1(x)                               # Conv1
        x = self.pool(self.leaky_relu(self.bn1(x)))     # MaxPool + Leaky ReLU + BatchNorm
        x = self.conv2(x)                               # Conv2
        x = self.pool(self.leaky_relu(self.bn2(x)))     # MaxPool + Leaky ReLU + BatchNorm
        x = self.conv3(x)                               # Conv3
        x = self.pool(self.leaky_relu(self.bn3(x)))     # MaxPool + Leaky ReLU + BatchNorm

        # Flatten the tensor for fully connected layers
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers
        x = self.fc1(x)                                 # FC1 (embedding layer)
 
        return x
    
