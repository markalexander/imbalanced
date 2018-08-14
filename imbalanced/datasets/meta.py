# -*- coding: utf-8 -*-

import numpy as np


class ImbalancedDatasetInfo:
    """Wrapper that provides info/statistics on the wrapped imbalanced dataset.

    Assumes binary imbalance (either regression or classification), for now.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self._minority_count = None

    @property
    def size(self):
        return len(self.dataset)

    @property
    def classes(self):
        return [0, 1]

    @property
    def inputs(self):
        return self.dataset.tensors[0].numpy()

    @property
    def targets(self):
        return self.dataset.tensors[1].numpy()

    @property
    def minority_targets(self):
        return self.targets[self.targets != 0]

    @property
    def majority_class(self):
        return 0

    @property
    def majority_count(self):
        return self.size - self.minority_count

    @property
    def minority_count(self):
        if self._minority_count is None:
            self._minority_count = np.count_nonzero(self.targets)
        return self._minority_count

    @property
    def majority_minority_ratio(self):
        if self.minority_count == 0:
            return None
        return self.majority_count / self.minority_count

    @property
    def minority_majority_ratio(self):
        if self.majority_count == 0:
            return None
        return self.minority_count / self.majority_count

    @property
    def target_density(self):
        hist, bin_edges = np.histogram(self.targets)
        return bin_edges.tolist(), hist.tolist()

    @property
    def minority_target_density(self):
        hist, bin_edges = np.histogram(self.minority_targets)
        return bin_edges.tolist(), hist.tolist()

    def summary(self):
        s = {
            'type': self.dataset.__class__.__name__,
            'size': self.size,
            'majority_count': self.majority_count,
            'minority_count': self.minority_count,
            'majority_minority_ratio': self.majority_minority_ratio,
            'minority_majority_ratio': self.minority_majority_ratio,
            'target_density': self.target_density,
            'minority_target_density': self.minority_target_density
        }
        return s

    def __repr__(self):
        return str(self.summary())
