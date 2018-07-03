# -*- coding: utf-8 -*-

"""
This file contains definitions for the various imbalancer classes.

Essentially, an imbalancer wraps an existing dataset and makes it (more)
imbalanced, through a variety of methods.  The new object can then be used as a
dataset in its own right.
"""

from abc import abstractmethod
from .datasets import Dataset, ResampledDataset


class Imbalancer:
    """Base class for all imbalancers."""

    @abstractmethod
    def imbalance(self, dataset: Dataset) -> ResampledDataset:
        pass
