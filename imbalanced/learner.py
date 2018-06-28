# -*- coding: utf-8 -*-

from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer


class LearningAlgorithm:
    """Defines (the key elements of) a learning algorithm."""

    def __init__(self, criterion, optimizer, patience):
        """Create a LearningAlgorithm object.

        :param criterion:  the optimization criterion
        :type  criterion:  object
        :param optimizer:  the optimizer
        :type  optimizer:  torch.optim.optimizer.Optimizer
        :param patience:   the number of epochs after which to terminate when
                           no improvement is seen in validation set performance
        :type  patience:   int
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
    def criterion(self):
        """Get the criterion.

        :return:  the criterion
        :rtype:   torch.nn.modules.loss._Loss
        """
        return self._criterion

    @criterion.setter
    def criterion(self, criterion):
        """Set the criterion.

        :param criterion:  the criterion
        :type  criterion:  torch.nn.modules.loss._Loss
        :return:           None
        """
        assert isinstance(criterion, _Loss),\
            ('The `criterion` argument must be an instance of '
             '`torch.nn.modules.loss._Loss`. '
             'Received `{}` instead.'.format(type(criterion)))
        self._criterion = criterion

    @property
    def optimizer(self):
        """Get the optimizer.

        :return:  the optimizer
        :rtype:   torch.optim.optimizer.Optimizer
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """Set the optimizer.

        :param optimizer:  the optimizer
        :type  optimizer:  torch.optim.optimizer.Optimizer
        :return:           None
        """
        assert isinstance(optimizer, Optimizer),\
            ('The `optimizer` argument must be an instance of '
             '`torch.optim.optimizer.Optimizer`. '
             'Received `{}` instead.'.format(type(optimizer)))
        self._optimizer = optimizer
