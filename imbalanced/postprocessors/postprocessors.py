# -*- coding: utf-8 -*-

"""
This file contains class definitions for the various post-processing methods.
"""

from abc import ABC, abstractmethod


class Postprocessor(ABC):

    @abstractmethod
    def __call__(self, inputs):
        pass
