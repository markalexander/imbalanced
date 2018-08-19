# -*- coding: utf-8 -*-

"""
Definitions for various binners.
"""

from typing import List, Tuple, Union
from abc import abstractmethod
import numpy as np

Num = Union[int, float]


class Binner:
    """Base class for all binners."""

    @abstractmethod
    def get_bins(self, dataset) -> List[Union[Tuple[Num], Tuple[Num, Num]]]:
        """Get the bin boundaries for the given dataset.

        Args:
            dataset: The dataset.

        Returns:
            The bin boundaries, as a list of 1- and 2-tuples.

        """
        pass

    def get_bin_classes(self, dataset) -> List[int]:
        """Get the class indices corresponding to the bins.

        Args:
            dataset: The dataset.

        Returns:
            The corresponding class indices.

        """
        return list(range(len(self.get_bins(dataset))))

    def get_bins_and_classes(self, dataset):
        return self.get_bins(dataset), self.get_bin_classes(dataset)


class ZeroNonZeroBinner(Binner):
    """Bins targets into zero and non-zero values."""

    def get_bins(self, dataset) -> List[Union[Tuple[Num], Tuple[Num, Num]]]:
        """Get the bin boundaries for the given dataset.

        Args:
            dataset: The dataset.

        Returns:
            The bin boundaries, as a list of 1- and 2-tuples.

        """
        return [(-np.inf, 0.), (0.,), (0., np.inf)]

    def get_bin_classes(self, dataset) -> List[int]:
        """Get the class indices corresponding to the bins.

        Args:
            dataset: The dataset.

        Returns:
            The corresponding class indices.

        """
        return [1, 0, 1]


class EqualBinner(Binner):
    """Bins targets into equally wide bins"""

    def get_bins(self, dataset) -> List[Union[Tuple[Num], Tuple[Num, Num]]]:
        """Get the bin boundaries for the given dataset.

        Args:
            dataset: The dataset.

        Returns:
            The bin boundaries, as a list of 1- and 2-tuples.

        """
        min_target = 0
        max_target = 1
        return [(0.,), (0., np.inf)]
