# -*- coding: utf-8 -*-

"""
This file contains the basic definition for an imbalanced data 'pipeline'.
"""


class ImbalancedDatasetProperties:

    def __init__(self, dataset):
        self._dataset = dataset
        self._imbalance_ratio = None
        self._class_overlap = None

    @property
    def imbalance_ratio(self):
        if self._imbalance_ratio is None:
            pass
        return self._imbalance_ratio
