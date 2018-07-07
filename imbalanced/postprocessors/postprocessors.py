# -*- coding: utf-8 -*-

"""
This file contains class definitions for the various post-processing methods.
"""

from abc import ABC, abstractmethod


class Postprocessor(ABC):

    @abstractmethod
    def train(self, original_dataset, predictions):
        pass

    @abstractmethod
    def process(self, inputs):
        pass

    def __call__(self, inputs):
        self.process(inputs)

    @abstractmethod
    def __repr__(self) -> str:
        """Get the canonical string representation for an instance of the
        object.

        Returns:
            The canonical string representation.

        """
        pass
