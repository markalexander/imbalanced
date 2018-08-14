# -*- coding: utf-8 -*-

"""
Definitions for various resamplers commonly used in imbalanced data scenarios.
"""

from typing import Any
from math import floor
import numpy as np
from torch.utils.data import Dataset, TensorDataset, SubsetRandomSampler


class IndexRandomSampler(SubsetRandomSampler):

    def __init__(self, indices) -> None:
        """Samples elements randomly from a given list of indices, without
        replacement.

        Essentially wraps PyTorch's SubsetRandomSampler with a class name
        that doesn't imply it must be used for subsets only, since indices
        may be freely used for super-sampling.
        """
        super().__init__(indices)


class RandomResampler(IndexRandomSampler):
    """Random resampler.

    Chooses resampled indices of the original dataset purely at random in
    proportion to the given rate.  Rate can be varied in order to provide
    both sub- and super-sampling.
    """

    def __init__(self, dataset: Dataset, rate: float) -> None:
        """Create an RandomResampler object.

        Args:
            dataset: The dataset to be resampled.
            rate:    The resampling rate.  Values 0 <= r < 1 result in under-
                     or subsampling of the data.  Value r > 1 result in over-
                     or super-sampling.  E.g. r = 0.3 will result in a dataset
                     with 30% of the data retained, and r = 1.1 will result in
                     the original data plus 10% of random repeats.

        Returns:
            None.

        """
        self.dataset = dataset
        self.rate = rate
        # Sample dataset indices
        super().__init__(
            resample_indices(np.arange(len(dataset)), rate)
        )


class RandomTargetedResampler(IndexRandomSampler):
    """Random resampling on a specific target value.

    Chooses resampled indices of the original dataset from the specified class
    in proportion to the given rate.  Rate can be varied in order to provide
    both sub- and super-sampling.
    """

    def __init__(self, dataset: TensorDataset,
                 resample_target_value: Any,
                 rate: float) -> None:
        """Create an RandomResampler object.

        Args:
            dataset:               The dataset to be resampled.
            resample_target_value: The target value of the class that will be
                                   resampled can e.g. be an integral class
                                   label, real-valued, or any possible target
                                   that will meet an equality (==) test.
            rate:                  The resampling rate.  Values 0 <= r < 1
                                   results in over-or super-sampling.  E.g.
                                   r = 0.3 will result in a dataset with 30% of
                                   the data retained, and r = 1.1 will result in
                                   the original data plus 10% of random repeats.

        Returns:
            None.

        """
        self.dataset = dataset
        self.rate = rate
        # Sample dataset indices
        targets = dataset.tensors[-1].numpy()
        # Grab indices of the class we want to resample, and those of the others
        resample_class_indices = np.where(targets == resample_target_value)[0]
        other_indices = np.where(targets != resample_target_value)[0]
        # Run the resampling on the class indices
        resample_class_indices = resample_indices(resample_class_indices, rate)
        # Concat both back into one big set of indices
        all_indices = np.concatenate((resample_class_indices, other_indices))
        # Shuffle it
        np.random.shuffle(all_indices)
        super().__init__(all_indices)


def resample_indices(all_indices, rate: float):
    """Randomly resample the given indices in proportion to the given rate.

    Args:
        all_indices: The indices to be resampled.
        rate:        The resampling rate.  Values 0 <= r < 1 result in
                     under- or subsampling of the data.  Value r > 1
                     result in over-or super-sampling.  E.g. r = 0.3 will
                     result in a dataset with 30% of the data retained,
                     and r = 1.1 will result in the original data plus
                     10% of random repeats.

    Returns:
        The indices that are part of the resample.

    """
    # Sample dataset indices
    assert rate >= 0., 'Resampling rate cannot be negative'
    # Start building sample
    sampled_indices = []
    # Shuffle the indices
    np.random.shuffle(all_indices)
    # What are we doing?
    if rate >= 1.0:
        # Super-sampling
        # Add the full copies as required
        full_copies = int(floor(rate))
        for i in range(full_copies):
            sampled_indices.extend(all_indices)
        # Reduce the remaining rate to be applied
        rate -= full_copies
    # Now rate should be strictly 0 <= rate < 1.0
    # Either because we reduced it in super-sampling, or it was
    # sub-sampling all along
    if rate > 0.0:
        target_len = int(round(rate * len(all_indices)))
        sampled_indices.extend(all_indices[:target_len])
    return np.array(sampled_indices)
