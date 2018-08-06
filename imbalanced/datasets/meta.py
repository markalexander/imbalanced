# -*- coding: utf-8 -*-

import numpy as np


class ImbalancedDatasetStats:

    def __init__(self, dataset):
        self.dataset = dataset
        targets = dataset.tensors[1].numpy()
        self.len = len(dataset)
        self.majority_count = len(np.where(targets == 0))
        self.minority_count = len(dataset) - self.majority_count
        self.majority_ratio = self.majority_count / self.minority_count
        self.minority_ratio = self.minority_count / self.majority_count

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        s = 'Total count: {}\n'.format(self.len)
        s += '\tmaj. count: {}, min. count: {}\n'.format(
            self.majority_count,
            self.minority_count,
        )
        s += '\tmaj ratio: {}, min. ratio: {}\n'.format(
            self.majority_ratio,
            self.minority_ratio
        )
        return s
