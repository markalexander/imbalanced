# -*- coding: utf-8 -*-

"""
Definitions for various binners.
"""

from abc import abstractmethod
import numpy as np


class Binner:
    """Base class for all binners."""

    @abstractmethod
    def get_bins(self, dataset):
        pass

    def get_bin_classes(self, dataset):
        return list(range(len(self.get_bins(dataset))))

    def get_bins_and_classes(self, dataset):
        return self.get_bins(dataset), self.get_bin_classes(dataset)


class ZeroNonZeroBinner(Binner):
    """Bins targets into zero and non-zero values.

    Assumes target is non-negative.
    """

    def get_bins(self, dataset):
        return [(-np.inf, 0.), (0.,), (0., np.inf)]

    def get_bin_classes(self, dataset):
        return [1, 0, 1]


class EqualBinner(Binner):
    """Bins targets into equally wide bins"""

    def get_bins(self, dataset):
        min_target = 0
        max_target = 1
        return [(0.,), (0., np.inf)]
