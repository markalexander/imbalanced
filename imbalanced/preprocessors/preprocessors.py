# -*- coding: utf-8 -*-

"""
This file contains class definitions for the various types of pre-processor.
"""

from abc import ABC, abstractmethod
import numpy as np
from ..datasets import ResampledDataset


class Preprocessor(ABC):
    """Pre-processor base class."""

    @abstractmethod
    def process(self, dataset):
        pass


class Resampler(Preprocessor):
    """Pre-processor base class."""

    def process(self, dataset):
        return self.resample(dataset)

    @abstractmethod
    def resample(self, dataset):
        pass
