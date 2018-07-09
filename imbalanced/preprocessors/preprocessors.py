# -*- coding: utf-8 -*-

"""
This file contains class definitions for the various types of pre-processor.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import numpy as np
from ..datasets import Dataset, ResampledDataset
from ..meta import CanonicalDictMixin


class Preprocessor(ABC, CanonicalDictMixin):
    """Pre-processor base class."""

    @abstractmethod
    def process(self, dataset: Dataset) -> Dataset:
        """Process a dataset according to the pre-processor.

        Args:
            dataset: The dataset to be processed.

        Returns:
            The processed dataset.
        """
        pass


class Resampler(Preprocessor):
    """Resampler base class."""

    def process(self, dataset: Dataset) -> Dataset:
        """Process a dataset according to the pre-processor.

        Args:
            dataset: The dataset to be processed.

        Returns:
            The processed dataset.
        """
        return self.resample(dataset)

    @abstractmethod
    def resample(self, dataset: Dataset) -> ResampledDataset:
        """Resample a dataset.

        Args:
            dataset: The dataset to be resampled.

        Returns:
            The resampled dataset.
        """
        pass


class RandomSubsampler(Resampler):
    """Purely random subsampler.

    Resamples all examples at random, regardless of class.
    """

    def __init__(self, rate: float) -> None:
        self.rate = rate

    def resample(self, dataset: Dataset) -> ResampledDataset:
        """Resample a dataset.

        Args:
            dataset: The dataset to be resampled.

        Returns:
            The resampled dataset.
        """
        original_len = len(dataset)
        target_len = int(round(self.rate * original_len))
        samples = np.random.randint(0, original_len, target_len)
        return ResampledDataset(dataset, samples)

    @property
    def args(self) -> List[Tuple[str, Any]]:
        """Get the canonical (ordered) list of arguments which define the
        current object.

        Returns:
            The arguments, as a list of tuples (arg_name, arg_value).

        """
        return [
            ('rate', self.rate)
        ]
