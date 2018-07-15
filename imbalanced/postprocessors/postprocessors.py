# -*- coding: utf-8 -*-

"""
This file contains class definitions for the various post-processing methods.
"""

from abc import ABC, abstractmethod
from ..meta import CanonicalArgsMixin


class Postprocessor(ABC, CanonicalArgsMixin):

    @abstractmethod
    def train(self, original_dataset, predictions):
        pass

    @abstractmethod
    def process(self, inputs):
        pass

    def __call__(self, inputs):
        self.process(inputs)
