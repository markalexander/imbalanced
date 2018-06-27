# -*- coding: utf-8 -*-

"""
This file contains definitions for the various imbalancer classes.

Essentially, an imbalancer wraps an existing dataset and makes it (more)
imbalanced, through a variety of methods.  The new object can then be used as a
dataset in its own right.
"""

from .datasets import DatasetWrapper


class Imbalancer(DatasetWrapper):
    """Base class for all imbalancers."""

    def __init__(self, dataset):
        super().__init__(dataset)
