# -*- coding: utf-8 -*-

"""
This file contains class definitions for the various types of pre-processor.
"""

from abc import abstractmethod
import numpy as np
from ..data.datasets import DatasetWrapper, SimpleDataset


class Preprocessor(DatasetWrapper):
    """Pre-processor base class."""

    def __init__(self, dataset):
        super().__init__(dataset)

    @abstractmethod
    def __getitem__(self, idx):
        """Get a data row by index.

        :param idx:  the index of the desired row
        :type  idx:  int
        :return:     the desired row, if it exists
        :rtype:      torch.Tensor
        """
        pass

    @abstractmethod
    def __len__(self):
        """Get the total number of rows in the dataset.

        :return:  the number of rows
        :rtype:   int
        """
        pass


class Resampler(DatasetWrapper):
    """Resampler base class.

    Wraps but does not copy the original dataset.  Instead stores only the
    minimum representation of the *differences* between the original dataset
    and the resampled one.  This means different things over sub- and
    supersampling:

        - For subsampling, we simply store the indices of those rows which are
          active in the subsample.  This is appopriate since, most commonly in
          the imbalanced data context, subsampling results in high sparsity of
          rows, and so storing this as e.g. a boolean mask on every row would be
          particularly wasteful.

        - For supersampling, there are two cases.  First, simple duplicates of
          original data points can simply be stored as indices referencing the
          corresponding row from the original data.  Second, methods which
          introduce entirely new data points (e.g. synthetic sampling) must
          store these new rows explicitly.

    Though those aspects could be abstracted to separate classes, some
    resampling methods could feasibly make use of both sub-, super- and
    synthetic sampling at the same time.

    Overall, this means there are two internal representation structures for
    storing a resampled dataset.  One ndarray to store indices of directly sub-
    and super-sampled data (i.e. allowing duplicates), and another dataset to
    store newly added rows.  This is somewhat inelegant, but allows for optimal
    memory and storage costs, and nonetheless this extra complexity is hidden
    from the end user.
    """

    def __init__(self, dataset=None):
        super().__init__(dataset)
        self._samples = None
        self._extra = None
        if dataset is not None:
            self.reset_and_resample()

    def reset_and_resample(self):
        self.reset()
        self.resample()

    def reset(self):
        """Reset the sampling to match the original data."""
        self._samples = np.arange(len(self.dataset), dtype=np.int)
        self._extra = SimpleDataset([], [])

    @abstractmethod
    def resample(self):
        """Run the resampling process and generate the internal
        representations.
        """
        pass

    def __getitem__(self, idx):
        """Get a data row by index.

        :param idx:  the index of the desired row
        :type  idx:  int
        :return:     the desired row, if it exists
        :rtype:      torch.Tensor
        """
        num_samples = len(self._samples)
        if idx < num_samples:
            return self.dataset[self._samples[idx]]
        else:
            return self._extra[num_samples + idx]

    def __len__(self):
        """Get the total number of rows in the dataset.

        :return:  the number of rows
        :rtype:   int
        """
        return len(self._samples) + len(self._extra)


class RandomSubsampler(Resampler):

    def __init__(self, dataset, rate):
        super().__init__(dataset)
        self.rate = rate

    def resample(self):
        self.reset()
        original_sample_count = len(self._samples)
        target_sample_count = int(round(self.rate * original_sample_count))
        self._samples = np.random.randint(0, original_sample_count,
                                          target_sample_count)
