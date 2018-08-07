# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Union, Any, Optional
import numpy as np
import torch
from torch.utils.data.dataset import Dataset, Subset


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
            start:   The start point of the partition.
            stop:    The stop point of the partition.

        Returns:
            None

        """
        super().__init__(dataset)
        assert isinstance(start, int), 'Partition start must be an integer.'
        assert isinstance(stop, int), 'Partition end must be an integer.'
        assert stop >= start, 'Partition stop must be >= start.'
        self.start = start
        self.stop = stop

    @property
    def tensors(self):
        """Get the (sliced) tensors for this partition.

        Assumes the wrapped dataset is an instance of TensorDataset.

        Returns:
            The tensors

        """
        return tuple([t for t in self.dataset.tensors])

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
            raise IndexError('Index %s is outside the partition' % idx)

    def __len__(self) -> int:
        """Get the total number of rows in the dataset.

        Returns:
            The number of rows.

        """
        return self.stop - self.start


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
        if sum(list(partition_spec.values())) <= 1:
            partition_spec = {name: round(fraction * len(dataset))
                              for name, fraction in partition_spec.items()}

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
            start = stop + 1
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
    def c_args(self) -> List[Tuple[str, Any]]:
        """Get the canonical (ordered) list of arguments ('c-args') which define
        the current object.

        Returns:
            The arguments, as a list of tuples (arg_name, arg_value).

        """
        return [
            ('dataset', self.dataset),
            ('partitions', self.partitions)
        ]


class ResampledDataset(Subset):
    """Resampled version of a wrapped dataset.

    For now, this is just a renaming of torch's Subset class--to better
    represent what it is used for here--with the addition of defaulting to
    a full sample of the underlying dataset.

    Note that a combination of synthetic generation and simple sampling can be
    achieved by using this wrapper on a ConcatDataset consisting of the
    original and the synthetically generated samples.  See the pre-processors
    module for more information.

    todo: seamlessly incorporate synthetic sampling via the above
    """

    def __init__(self, dataset: Dataset,
                 indices: Optional[np.ndarray] = None) -> None:
        if indices is None:
            # Default to entire dataset
            indices = np.arange(len(dataset), dtype=np.int)
        super().__init__(dataset, indices)
