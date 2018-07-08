# -*- coding: utf-8 -*-

"""
This file contains class definitions for the various post-processing methods.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from ..misc import CanonicalDictMixin


class Postprocessor(ABC, CanonicalDictMixin):

    @abstractmethod
    def train(self, original_dataset, predictions):
        pass

    @abstractmethod
    def process(self, inputs):
        pass

    def __call__(self, inputs):
        self.process(inputs)

    @property
    @abstractmethod
    def cdict(self) -> OrderedDict:
        """Get the canonical dict representation of the current object.

        Returns:
            The canonical dict representation.

        """
        pass
