# -*- coding: utf-8 -*-

from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer


class LearningAlgorithm:
    """Defines (the key elements of) a learning algorithm."""

    def __init__(self, criterion: _Loss,
                 optimizer: Optimizer, patience: int = 5) -> None:
        """Create a LearningAlgorithm object.

        Args:
            criterion: The optimization criterion.
            optimizer: The optimizer.
            patience:  The number of epochs after which to terminate when no
                       improvement is seen in validation set performance.

        """
        # Init
        self._criterion = None
        self._optimizer = None
        self._patience = None
        # Set
        self.criterion = criterion
        self.optimizer = optimizer
        self.patience = patience

    @property
    def criterion(self) -> _Loss:
        """Get the criterion.

        Returns:
            The criterion.

        """
        return self._criterion

    @criterion.setter
    def criterion(self, criterion: _Loss) -> None:
        """Set the criterion.

        Args:
            criterion: The criterion to be set.

        """
        assert isinstance(criterion, _Loss),\
            ('The `criterion` argument must be an instance of '
             '`torch.nn.modules.loss._Loss`. '
             'Received `{}` instead.'.format(type(criterion)))
        self._criterion = criterion

    @property
    def optimizer(self) -> Optimizer:
        """Get the optimizer.

        Returns:
            The optimizer.
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        """Set the optimizer.

        Args:
            optimizer: The optimizer to be set

        """
        assert isinstance(optimizer, Optimizer),\
            ('The `optimizer` argument must be an instance of '
             '`torch.optim.optimizer.Optimizer`. '
             'Received `{}` instead.'.format(type(optimizer)))
        self._optimizer = optimizer

    @property
    def patience(self) -> int:
        """Get the patience value.

        Returns:
            The patience value.
        """
        return self._patience

    @patience.setter
    def patience(self, patience: int):
        """Set the patience value.

        Args:
            patience: The patience value to be set.

        """
        assert int(patience) == patience,\
            ('The `patience` argument must be an integer. '
             'Received `{}` instead.'.format(type(patience)))
        self._patience = patience
