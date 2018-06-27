# -*- coding: utf-8 -*-

"""
This file contains definitions for the various imbalancer classes.

Essentially, an imbalancer wraps an existing dataset and makes it (more)
imbalanced, through a variety of methods.  The new object can then be used as a
dataset in its own right.
"""

from .datasets import Dataset


class Imbalancer(Dataset):
    """Base class for all imbalancers."""

    def __init__(self, dataset):
        self._dataset = None
        self.dataset = dataset

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        assert isinstance(dataset, Dataset),\
            'Wrapped object must be an instance of Dataset (or a subclass).'
        self._dataset = dataset


