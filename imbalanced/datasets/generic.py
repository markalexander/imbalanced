# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Union, Any, Optional
from math import ceil
import numpy as np
import torch
from torch.utils.data import Dataset
from ..samplers import IndexSampler
from ..binners import Binner


class DatasetWrapper(Dataset):
    """Base class for datasets that result from wrapping another Dataset object.
    """

    def __init__(self, dataset: Dataset,
                 lock_dataset: bool = True) -> None:
        """Create a DatasetWrapper object.

        Args:
            dataset:      The dataset to be wrapped.
            lock_dataset: Whether the wrapped dataset can be updated after
                          the wrapper object is constructed.  Typically set
                          to True, since doing so is likely to break indexing
                          used by the wrapper.
        """
        # Init
        self._dataset = None
        # Set
        self.dataset = dataset
        self._lock_dataset = True
        if isinstance(lock_dataset, bool):
            self._lock_dataset = lock_dataset

    @property
    def dataset(self) -> Dataset:
        """Get the wrapped dataset.

        Returns:
            The wrapped dataset.

        """
        return self._dataset

    @dataset.setter
    def dataset(self, dataset) -> None:
        """Set the wrapped dataset.

        Args:
            dataset: The dataset to be wrapped.

        """
        if self._dataset is not None:
            assert not self._lock_dataset,\
                'Wrapped dataset cannot be changed ' \
                'after wrapper has been initialized.'
        assert isinstance(dataset, Dataset),\
            'Wrapped object must be an instance of Dataset (or a subclass).'
        self._dataset = dataset

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a data row by index.

        Default to passing off to the wrapped dataset.

        Args:
            idx: The index of the desired row.

        Returns:
            The desired row (input, target)

        """
        return self.dataset[idx]

    def __len__(self) -> int:
        """Get the total number of rows in the dataset.

        Default to passing off to the wrapped dataset.

        Returns:
            The number of rows.

        """
        return self.dataset.__len__()


class DatasetPartition(DatasetWrapper):
    """Single partition of a dataset."""

    def __init__(self, dataset: Dataset, start: int, stop: int) -> None:
        """Create a DatasetPartition object.

        Indexing (for start and end values) is consistent with Python's slice().
        I.e. counting starts at zero, start value is inclusive, and
        stop value is non-inclusive.  So, for example, start = 1, stop = 4 would
        include the 1st, 2nd, and 3rd rows of the wrapped dataset.

        Args:
            dataset: The dataset that this is a partition of.
            start:   The start point of the partition.  Inclusive.
            stop:    The stop point of the partition.  Exclusive.

        Returns:
            None

        """
        super().__init__(dataset)
        assert isinstance(start, int), 'Partition start must be an integer.'
        assert isinstance(stop, int), 'Partition end must be an integer.'
        assert stop <= len(dataset),\
            'Partition stop must not exceed length of dataset.'
        assert stop >= start, 'Partition stop must be >= start.'
        self.start = start
        self.stop = stop

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a data row by index.

        Args:
            idx: The index of the desired row.

        Returns:
            The desired row (input, target)

        """
        if 0 <= idx < len(self):
            return self.dataset[self.start + idx]
        else:
            raise IndexError('Index {} is outside the partition ({})'.format(
                idx,
                len(self)
            ))

    def __len__(self) -> int:
        """Get the total number of rows in the dataset.

        Returns:
            The number of rows.

        """
        return self.stop - self.start

    @property
    def args(self) -> List[Tuple[str, Any]]:
        """Get the canonical (ordered) list of arguments which define the
        current object.

        Returns:
            The arguments, as a list of tuples (arg_name, arg_value).

        """
        return [
            ('dataset', self.dataset),
            ('start', self.start),
            ('stop', self.stop)
        ]


class PartitionedDataset(DatasetWrapper):
    """Partitioned dataset wrapper.

    Allows for e.g. partitioning a given dataset into train/val/test splits.
    """

    def __init__(
        self,
        dataset: Dataset,
        partition_spec: Optional[Dict[str, Union[int, float]]] = None
    ) -> None:
        """Create a PartitionedDataset object.

        Partition a dataset into the given partitions.

        Args:
            dataset:        The dataset to be partitioned.
            partition_spec: The partition specification.  A dictionary of partition
                            names and sizes in the form
                            {'partition1': size1, 'partition2': size2}

        Returns:
            None

        """

        super().__init__(dataset)

        # Reasonable defaults
        # Equivalent to 80:20 split twice
        if partition_spec is None:
            partition_spec = {'train': 0.64, 'val': 0.16, 'test': 0.2}

        # Basic validation
        assert isinstance(partition_spec, dict),\
            'Partition specification must be a dictionary object'
        assert len(partition_spec) >= 2, \
            'Partition specification must define at least two partitions'
        assert min(partition_spec.values()) > 0, \
            'Partitions must be non-empty'

        # Convert fractional partitions to integer sizes
        if sum(partition_spec.values()) <= 1:
            int_partition_spec = {}
            n_partitions = len(partition_spec)
            for name, fraction in partition_spec.items():
                if len(int_partition_spec) == n_partitions - 1:
                    # Last partition, give remaining len if this is smaller than
                    # the calculated partition
                    int_partition_spec[name] = min(
                        int(ceil(fraction * len(dataset))),
                        len(dataset) - sum(int_partition_spec.values())
                    )
                else:
                    int_partition_spec[name] = int(
                        round(fraction * len(dataset)))
            partition_spec = int_partition_spec

        # Further validation
        for size in partition_spec.values():
            assert size == int(size),\
                'Partition sizes must be integer, or fractional and sum to <= 1'
        assert sum(partition_spec.values()) == len(dataset),\
            'Partition sizes must sum to dataset size'

        # Build partitions
        partitions = {}
        start = 0
        for name in list(partition_spec.keys()):
            stop = start + partition_spec[name]
            partitions[name] = DatasetPartition(dataset, start, stop)
            start = stop
        self.partitions = partitions

    def __getattr__(self, name):
        """Get the partition with the given name.

        Allows for shortcuts like dataset.test

        Args:
            name: The name of the partition to get.

        Returns:
            The requested partition.

        """
        if name in self.partitions:
            return self.partitions[name]
        else:
            raise AttributeError()

    @property
    def args(self) -> List[Tuple[str, Any]]:
        """Get the canonical (ordered) list of arguments which define the
        current object.

        Returns:
            The arguments, as a list of tuples (arg_name, arg_value).

        """
        return [
            ('dataset', self.dataset),
            ('partitions', self.partitions)
        ]

    def __repr__(self) -> str:
        """Get a string representation of the current object.

        Returns:
            The string representation.

        """
        return '<{}({})>'.format(self.__class__.__name__, self.args)


class ResampledDataset(DatasetWrapper):
    """Resampled version of a dataset, using sample indices."""

    def __init__(self, dataset: Dataset, indices) -> None:
        super().__init__(dataset)
        if isinstance(indices, IndexSampler):
            indices = indices.get_sample_indices(dataset)
        self.indices = indices

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a data row by index.

        Args:
            idx: The index of the desired row.

        Returns:
            The desired row (input, target)

        """
        return self.dataset[self.indices[idx]]

    def __len__(self) -> int:
        """Get the total number of rows in the dataset.

        Returns:
            The number of rows.

        """
        return len(self.indices)


class BinnedDataset(DatasetWrapper):
    """Binned dataset.

    Groups specified intervals (bins) of the target values into classes.

    Can be used on real-valued or class labelled data (to merge classes).
    """

    def __init__(self, dataset: Dataset, bins, bin_classes=None) -> None:
        """Create a BinnedDataset object.

        todo: allow more flexibility in interval specification.  This is not
        deeply important for the current typical use cases, though it could
        crop up where e.g. we need wholly-closed or wholly-open intervals.

        Args:
            dataset: The dataset to be binned.
            bins:    The bins, as ordered intervals of the form [a, b), i.e.
                     with a inclusive and b exclusive.  Singletons may also be
                     given for an exact match.  Where a singleton is given,
                     the next interval is treated as (a, b).
        """
        super().__init__(dataset)
        if isinstance(bins, Binner):
            bins, bin_classes = bins.get_bins_and_classes(dataset)
        if bin_classes is None:
            bin_classes = list(range(len(bins)))
        self.bins = bins
        self.bin_classes = bin_classes
        self.idx_bins = -1 * np.ones((len(dataset),), dtype=np.int)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a data row by index.

        Args:
            idx: The index of the desired row.

        Returns:
            The desired row (input, target)

        """
        row = self.dataset[idx]
        if self.idx_bins[idx] == -1:
            found_bin = False
            for class_idx, bin in enumerate(self.bins):
                if row[-1] < bin[0]:
                    # Should have been before this bin, so can't be in any bin
                    break
                elif len(bin) == 1:
                    if bin[0] == row[-1]:
                        # Singleton
                        self.idx_bins[idx] = self.bin_classes[class_idx]
                        found_bin = True
                        break
                elif bin[0] <= row[-1] < bin[1]:
                    # Half open interval
                    self.idx_bins[idx] = self.bin_classes[class_idx]
                    found_bin = True
                    break
            if not found_bin:
                raise ValueError('Target value {} does not belong to any bin'.format(
                    row[-1]
                ))
        return row[:-1] + (torch.LongTensor([self.idx_bins[idx]]),)


class TransformedDataset(DatasetWrapper):
    """Transformed dataset wrapper.

    Apply the transform to the data, e.g. a predictor or other transformation.
    """

    def __init__(self, dataset: Dataset, transform) -> None:
        """Create a TransformedDataset object.

        Args:
            dataset:   The dataset to wrap.
            transform: The transform to be applied.

        Returns:
              None

        """
        super().__init__(dataset)
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a data row by index.

        Args:
            idx: The index of the desired row.

        Returns:
            The desired row (input, target)

        """
        row = self.dataset[idx]
        for i in range(len(row) - 1):
            row[i] = self.transform(row[i])
        return row
