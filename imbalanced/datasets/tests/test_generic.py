# -*- coding: utf-8 -*-

import pytest
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import TensorDataset
from typing import Tuple, Optional, Union, Dict
from ..generic import DatasetWrapper, DatasetPartition, PartitionedDataset,\
    ResampledDataset


def dataset_rows_are_equal(row1: Tuple[Tensor, ...],
                           row2: Tuple[Tensor, ...]) -> bool:
    """Utility function for comparing rows of form (tensor1, tensor2, ...),
    e.g. as returned by __getitem__ on a TensorDataset.

    Args:
        row1: The first row.
        row2: The second row.

    Returns:
        Whether the rows are equal or not.
    """
    return all((row1[i] == row2[i]).all() for i in range(len(row1)))


def make_random_dataset(size: int = 100,
                        input_dim: int = 5,
                        target_dim: int = 5) -> Tuple[TensorDataset,
                                                      np.ndarray, np.ndarray]:
    """Create a random TensorDataset object.

    Args:
        size:       The desired number of rows
        input_dim:  The desired input dimension.
        target_dim: The desired target dimension.

    Returns:
        The random TensorDataset object.

    """
    inputs = torch.rand(size, input_dim)
    targets = torch.rand(size, target_dim)
    dataset = TensorDataset(inputs, targets)
    return dataset, inputs, targets


class TestDatasetWrapper:
    """Tests for the DatasetWrapper class."""

    def test_wrapped_dataset_locking(self) -> None:
        """Test the wrapped dataset locking behaviour."""
        # Locking on by default
        wrapper = self.make_random(10)
        with pytest.raises(AssertionError):
            wrapper.dataset = make_random_dataset()[0]
        # Locking off
        wrapper = self.make_random(10, lock_dataset=False)
        wrapper.dataset = make_random_dataset()[0]

    @staticmethod
    def make_random(size: Optional[int] = None,
                    *args, **kwargs) -> DatasetWrapper:
        """Create a random DatasetWrapper-based object.

        Args:
            size:     The desired number of rows in the random dataset.
            *args:    Variable length argument list to be passed to the
                      DatasetWrapper-based object's constructor.
            **kwargs: Arbitrary keyword arguments to be passed to the
                      DatasetWrapper-based object's constructor.

        Returns:
            The random DatasetWrapper-based object.

        """
        class Wrapper(DatasetWrapper):
            def __init__(self, *args_, **kwargs_) -> None:
                super().__init__(*args_, **kwargs_)

        return Wrapper(make_random_dataset(size)[0], *args, **kwargs)


class TestDatasetPartition:
    """Tests for the DatasetPartition class."""

    def test_indexes_correctly(self):
        dataset = make_random_dataset(10)[0]
        partition = DatasetPartition(dataset, 1, 5)
        assert len(partition) == 4
        assert dataset_rows_are_equal(partition[0], dataset[1])
        assert dataset_rows_are_equal(partition[1], dataset[2])
        assert dataset_rows_are_equal(partition[2], dataset[3])
        assert dataset_rows_are_equal(partition[3], dataset[4])


class TestPartitionedDataset:
    """Tests for the PartitionedDataset class."""

    def test_integer_partitions(self) -> None:
        """Test whether the constructor accepts and correctly indexes partitions
        defined as integer sizes.
        """
        dataset = self.make_random(10, {'a': 5, 'b': 3, 'c': 2})
        # Correct len
        assert len(dataset.a) == 5
        assert len(dataset.b) == 3
        assert len(dataset.c) == 2
        assert len(dataset.a) + len(dataset.b) + len(dataset.c) == len(dataset)
        # No overlapping partitions
        ps = [
            (dataset.a.start, dataset.a.stop),
            (dataset.b.start, dataset.b.stop),
            (dataset.c.start, dataset.c.stop)
        ]
        for i in range(len(ps)):
            for j in range(len(ps)):
                if i != j:
                    assert not ps[i][0] <= ps[j][0] <= ps[i][1]
                    assert not ps[j][0] <= ps[i][0] <= ps[j][1]

    def test_fractional_partitions(self) -> None:
        """Test whether the constructor accepts and correctly indexes partitions
        defined as fractional sizes.
        """
        dataset = self.make_random(100, {'a': 0.5, 'b': 0.3, 'c': 0.2})
        assert len(dataset.a) == 50
        assert len(dataset.b) == 30
        assert len(dataset.c) == 20

    def test_tied_partitions(self) -> None:
        """Test whether the constructor correctly splits a dataset when rounding
        issues cause a total length that is not correct.

        E.g. dataset with 5 items split into halves.  Each half will get
        0.5 * 5 = 2.5 of the dataset.  When rounded this means two partitions
        of 3 elements.  But that gives a total len of 6, which is more than the
        original.
        """
        dataset = self.make_random(5, {'a': 0.5, 'b': 0.5})
        assert \
            len(dataset.a) == 3 and len(dataset.b) == 2 \
            or len(dataset.a) == 2 and len(dataset.b) == 3
        dataset = self.make_random(10, {'a': 1/3, 'b': 1/3, 'c': 1/3})
        assert any((
            len(dataset.a) == 4 and len(dataset.b) == 3 and len(dataset.c) == 3,
            len(dataset.a) == 3 and len(dataset.b) == 4 and len(dataset.c) == 3,
            len(dataset.a) == 3 and len(dataset.b) == 3 and len(dataset.c) == 4
        ))

    def test_default_partition_spec(self) -> None:
        """Test whether the default partition spec is applied when none are
        specified.
        """
        dataset = self.make_random(100)
        assert len(dataset.train) == 64
        assert len(dataset.val) == 16
        assert len(dataset.test) == 20

    def test_rejects_invalid_partitions(self) -> None:
        """Test whether invalid partition structures are rejected."""
        # Not a dict
        with pytest.raises(AssertionError):
            # noinspection PyTypeChecker
            self.make_random(10, ['a', 'b', 'c'])
        # All zero-sized partitions
        with pytest.raises(AssertionError):
            self.make_random(15, {'a': 0, 'b': 0, 'c': 0})
        # Correct overall size but a zero-sized partition
        with pytest.raises(AssertionError):
            self.make_random(10, {'a': 5, 'b': 5, 'c': 0})
        # Negative partition sizes
        with pytest.raises(AssertionError):
            self.make_random(10, {'a': 5, 'b': 5, 'c': -5})
        # Partitions greater than total length
        with pytest.raises(AssertionError):
            self.make_random(10, {'a': 5, 'b': 5, 'c': 5})
        # Partitions less than total length
        with pytest.raises(AssertionError):
            self.make_random(10, {'a': 3, 'b': 3, 'c': 3})
        # Fractional partitions sum to > 1
        with pytest.raises(AssertionError):
            self.make_random(10, {'a': 0.5, 'b': 0.5, 'c': 0.5})
        # Fractional partitions sum to < 1
        with pytest.raises(AssertionError):
            self.make_random(10, {'a': 0.3, 'b': 0.3, 'c': 0.3})

    def test_rejects_invalid_partition_selection(self) -> None:
        """Test whether invalid partition selections are rejected."""
        dataset = self.make_random(10, {'a': 5, 'b': 5})
        with pytest.raises(AttributeError):
            assert len(dataset.c) > 0

    @staticmethod
    def make_random(size: int = 100,
                    partitions: Optional[Dict[str, Union[int, float]]] = None)\
            -> PartitionedDataset:
        """Create a random PartitionedDataset object.

        Args:
            size:       The desired number of rows in the random dataset.
            partitions: The partitions specification to be passed to
                        PartitionedDataset's constructor.

        Returns:
            The random SimpleDataset object.

        """
        return PartitionedDataset(
            make_random_dataset(size)[0],
            partitions
        )


class TestResampledDataset:
    """Tests for the ResampledDataset class."""

    @staticmethod
    def make_random(size: int = 100,
                    samples: Optional[np.ndarray] = None)\
            -> Tuple[ResampledDataset, TensorDataset]:
        """Create a random ResampledDataset object.

        Args:
            size:    The desired number of rows in the random dataset.
            samples: The desired samples to use in the object.

        Returns:
            The random ResampledDataset.

        """
        original = make_random_dataset(size)[0]
        resampled = ResampledDataset(original, samples)
        return resampled, original
