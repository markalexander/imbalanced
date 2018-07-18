# -*- coding: utf-8 -*-

import pytest
import numpy as np
import torch
from torch import random, Tensor
from torch.utils.data import TensorDataset
from typing import Tuple, Optional, Union, Dict
from ..generic import DatasetWrapper, PartitionedDataset, ResampledDataset


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
    return all(row1[i] == row2[i] for i in range(len(row1)))


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

    def test_repr(self) -> None:
        """Test the canonical string representation."""
        # todo
        pass

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


class TestPartitionedDataset:
    """Tests for the PartitionedDataset class."""

    def test_integer_partitions(self) -> None:
        """Test whether the constructor accepts and correctly indexes partitions
        defined as integer sizes.
        """
        dataset = self.make_random(10, {'a': 5, 'b': 3, 'c': 2})
        # Test partition lengths
        dataset.set_active_partition('a')
        assert len(dataset) == 5
        dataset.set_active_partition('b')
        assert len(dataset) == 3
        dataset.set_active_partition('c')
        assert len(dataset) == 2

    def test_fractional_partitions(self) -> None:
        """Test whether the constructor accepts and correctly indexes partitions
        defined as fractional sizes.
        """
        dataset = self.make_random(100, {'a': 0.6, 'b': 0.2, 'c': 0.2})
        dataset.set_active_partition('a')
        assert len(dataset) == 60
        dataset.set_active_partition('b')
        assert len(dataset) == 20
        dataset.set_active_partition('c')
        assert len(dataset) == 20

    def test_partition_shortcuts(self) -> None:
        """Test the use of shortcut properties to train/val/test."""
        dataset = self.make_random(10, {'train': 5, 'val': 3, 'test': 2})
        dataset = dataset.train
        assert dataset.active_partition == 'train'
        dataset = dataset.val
        assert dataset.active_partition == 'val'
        dataset = dataset.test
        assert dataset.active_partition == 'test'

    def test_default_partitions(self) -> None:
        """Test whether the default partitions are applied when none are
        specified.
        """
        dataset = self.make_random(100)
        dataset.set_active_partition('train')
        assert len(dataset) == 64
        dataset.set_active_partition('val')
        assert len(dataset) == 16
        dataset.set_active_partition('test')
        assert len(dataset) == 20

    def test_default_active_partition(self) -> None:
        """Test whether the default active partition is set appropriately."""
        # Defaults to train, if it exists
        dataset = self.make_random(10, {'train': 5, 'test': 5})
        assert dataset.active_partition == 'train'
        # Defaults to one of the given partitions otherwise
        dataset = self.make_random(10, {'a': 5, 'b': 5})
        assert dataset.active_partition in ['a', 'b']

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
        dataset.set_active_partition('a')
        # Non-existent partition
        with pytest.raises(AssertionError):
            dataset.set_active_partition('does_not_exist')
        # Check that didn't change the active partition
        assert dataset.active_partition == 'a'

    def test_rejects_invalid_indexing(self) -> None:
        """Test whether invalid partition indexing is rejected."""
        dataset = self.make_random(10, {'a': 5, 'b': 3, 'c': 2})
        # Selecting data beyond the end of the whole dataset
        dataset.set_active_partition('a')
        with pytest.raises(IndexError):
            _ = dataset[11]
        # Selecting data beyond the end of the active partition
        dataset.set_active_partition('a')
        with pytest.raises(IndexError):
            _ = dataset[5]
        dataset.set_active_partition('b')
        with pytest.raises(IndexError):
            _ = dataset[3]
        dataset.set_active_partition('c')
        with pytest.raises(IndexError):
            _ = dataset[2]

    def test_repr(self) -> None:
        """Test the canonical string representation."""
        # todo
        pass

    @staticmethod
    def make_random(size: Optional[int] = None,
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

    def test_masked_indexes_correctly(self) -> None:
        """Test whether the masked indexing/length works correctly."""
        samples = np.array([1, 2, 5, 7], dtype=np.int)
        resampled, original = self.make_random(10, samples)
        assert len(resampled) == 4
        i_resampled = 0
        for i_orig in range(len(original)):
            if i_orig in samples:
                assert dataset_rows_are_equal(original[i_orig],
                                              resampled[i_resampled])
                i_resampled += 1

    def test_all_rows_sampled_by_default(self) -> None:
        """Test whether the default behaviour (no provided samples) works
        correctly."""
        resampled, original = self.make_random()
        assert len(resampled) == len(original)
        for i in range(len(original)):
            assert dataset_rows_are_equal(original[i], resampled[i])

    def test_rejects_invalid_samples(self) -> None:
        """Test whether invalid input is rejected."""
        # Not an ndarray
        with pytest.raises(AssertionError):
            # noinspection PyTypeChecker
            self.make_random(samples=[1, 2])
        # Invalid shape
        with pytest.raises(AssertionError):
            self.make_random(2, np.array([[1, 2], [3, 4]], dtype=np.int))
        # Contains indices beyond end of dataset
        with pytest.raises(AssertionError):
            self.make_random(10, np.array([1, 2, 11], dtype=np.int))
        # Contains negative indices
        with pytest.raises(AssertionError):
            self.make_random(10, np.array([-1, 2, 1], dtype=np.int))

    def test_repr(self) -> None:
        """Test the canonical string representation."""
        # todo
        pass

    @staticmethod
    def make_random(size: Optional[int] = None,
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
