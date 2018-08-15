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


class ZeroPositiveBinner:
    """Bins targets into zero and > 0 values.

    Assumes target is non-negative.
    """

    def get_bins(self, dataset):
        return [(0.,), (0., np.inf)]
