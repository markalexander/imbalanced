# -*- coding: utf-8 -*-

"""
This file contains class definitions for the various types of pre-processor.
"""

from abc import abstractmethod
from ..data.datasets import DatasetWrapper


class Preprocessor(DatasetWrapper):
    """Pre-processor base class."""

    def __init__(self, dataset):
        super().__init__(dataset)

    @abstractmethod
    def __getitem__(self, idx):
        """Get a data row by index.

        :param idx:  the index of the desired row
        :type  idx:  int
        :return:     the desired row, if it exists
        :rtype:      torch.Tensor
        """
        pass

    @abstractmethod
    def __len__(self):
        """Get the total number of rows in the dataset.

        :return:  the number of rows
        :rtype:   int
        """
        pass


class RandomSubsampler(Preprocessor):

    def __init__(self, dataset, rate):
        super().__init__(dataset)
        self.rate = rate


class RandomNegativeSubsampler(Preprocessor):
    pass
