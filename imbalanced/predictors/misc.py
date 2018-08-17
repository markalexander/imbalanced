# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from abc import ABC, abstractmethod
from typing import Optional
import torch.nn.functional as F
from imbalanced.datasets import BinnedDataset, TransformedDataset, ResampledDataset
from imbalanced.samplers import RandomTargetedResampler
from imbalanced.binners import ZeroNonZeroBinner, EqualBinner
from torch.nn.init import xavier_uniform_
from torch.nn import Linear


class TrainableModelMixin(ABC):

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def train_all(self, train_dataset: Dataset, val_dataset: Dataset) -> None:
        pass


class FeedForwardBase(nn.Module, TrainableModelMixin):
    """Simple multi-layer feed-forward network."""

    def __init__(self, in_dim: int, out_dim: int, depth: int,
                 hidden_dim: Optional[int] = None) -> None:
        """Create a FeedForwardRegressor object.

        Args:
            in_dim:     The dimension of the input.
            out_dim:    The dimension of the output.  Assumes classification if
                        out_dim > 1.
            depth:      The number of hidden layers.
            hidden_dim: The dimension of the hidden layers.  Defaults to
                        midpoint of in_dim and out_dim.

        Returns:
            None.

        """
        if hidden_dim is None:
            hidden_dim = int(round((in_dim + out_dim) / 2))
        super().__init__()
        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        ]
        for i in range(depth):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ])
        layers.append(
            nn.Linear(hidden_dim, out_dim)
        )
        self.seq = nn.Sequential(*layers)

    def train_all(self, train_dataset: Dataset, val_dataset: Dataset) -> None:
        """Train the model with the given data.

        Convenience method with reasonable defaults.  Components may be trained
        directly with a custom/external implementation, if desired.

        Args:
            train_dataset: The training dataset.
            val_dataset:   The validation dataset.

        Returns:
            None.

        """
        train_nn(self, train_dataset, val_dataset)


def train_nn(net, train_dataset: Dataset, val_dataset: Dataset,
             optimizer=None, criterion=None, patience=5, is_classification=False) -> None:
    """Train a neural net module with the given data.

    Convenience method with reasonable defaults.  Components may be trained
    directly with a custom/external implementation, if desired.

    Args:
        train_dataset: The training dataset.
        val_dataset:   The validation dataset.
        optimizer:     The optimizer object.
        criterion:     The criterion for optimization.
        patience:      The patience value for terminating training.

    Returns:
        None.

    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Training {}, classification: {}'.format(net, is_classification))

    if optimizer is None:
        optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

    if criterion is None:
        if is_classification:
            criterion = torch.nn.NLLLoss().to(device)
            crit_name = 'NLL Loss'
        else:
            criterion = torch.nn.MSELoss().to(device)
            crit_name = 'MSE'

    # Move to CUDA
    net.to(device)

    # Initialize weights
    net.apply(xavier_uniform_init)

    # Dataloaders
    # params = {'batch_size': 64, 'num_workers': 4}
    train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=10, num_workers=4)

    # Start main training loop
    epoch = 0
    val_crit = []
    while not meets_patience_termination_cond(val_crit, patience):

        # Loop through training batches
        for inputs, targets in train_dataloader:

            # Reset gradients
            optimizer.zero_grad()
            net.zero_grad()

            # Variables
            inputs = Variable(inputs).to(device)
            targets = Variable(targets).to(device)

            # Forward pass
            outputs = net(inputs)
            if is_classification:
                targets = targets.squeeze()
            loss = criterion(outputs, targets)

            # Backward pass/update
            loss.backward()
            optimizer.step()

        # Epoch complete
        epoch += 1

        # Check val criterion
        count = 0
        total_loss = 0
        for inputs, targets in val_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if is_classification:
                targets = targets.squeeze()
            outputs = net(inputs)
            total_loss += criterion(outputs, targets)
            count += 1
        val_crit.append(total_loss / count)
        print('{{"metric": "Epoch val {}", "value": {}}}'.format(crit_name, val_crit[-1]))


def meets_patience_termination_cond(val_crit, patience, patience_rel_tol=0.001):
    return len(val_crit) > patience\
        and val_crit[-1] * (1 + patience_rel_tol) >= val_crit[-patience]


def xavier_uniform_init(m):
    """Xavier weight initializer.

    Args:
        m: A PyTorch module, typically a layer.

    Returns:
        None

    """
    if isinstance(m, Linear):
        xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
