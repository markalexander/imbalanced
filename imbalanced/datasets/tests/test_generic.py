# -*- coding: utf-8 -*-

import pytest
import numpy as np
import torch
from typing import Tuple, Optional, Union, Dict, List
from ..generic import Dataset, SimpleDataset, DatasetWrapper,\
    PartitionedDataset, ConcatenatedDataset, ResampledDataset


def dataset_rows_are_equal(row1: Tuple[torch.Tensor, torch.Tensor],
                           row2: Tuple[torch.Tensor, torch.Tensor]) -> bool:
    """Utility function for comparing rows of form (inputs, outputs),
    as returned by __getitem__ on our datasets."""
    return (row1[0] == row2[0]).all() and (row1[1] == row2[1]).all()


class TestSimpleDataset:
    """Tests for the SimpleDataset class."""

    def test_indexes_numpy_ndarray(self) -> None:
        """Test whether the constructor accepts and correctly indexes numpy
        ndarray objects.
        """
        dataset, inputs, targets = self.make_random()
        # Check length is correct
        assert len(dataset) == len(inputs) == len(targets)
        # Check all rows are equal
        for i in range(len(inputs)):
            assert (dataset[i][0].numpy() == inputs[i]).all()
            assert (dataset[i][1].numpy() == targets[i]).all()

    def test_indexes_list_of_lists(self) -> None:
        """Test whether the constructor accepts and correctly indexes
        lists-of-lists.
        """
        _, inputs, targets = self.make_random()
        dataset = SimpleDataset(list(inputs.tolist()), list(targets.tolist()))
        # Check length is correct
        assert len(dataset) == len(inputs) == len(targets)
        # Check all rows are equal
        for i in range(len(inputs)):
            assert (dataset[i][0].numpy() == np.array(inputs[i])).all()
            assert (dataset[i][1].numpy() == np.array(targets[i])).all()

    def test_rejects_length_mismatch(self) -> None:
        """Test whether the constructor raises an error for inputs and targets
        with differing lengths.
        """
        with pytest.raises(AssertionError):
            SimpleDataset(
                np.ones((5, 5)),
                np.ones((6, 5))
            )

    def test_repr(self) -> None:
        """Test the canonical string representation."""
        # d = self.make_random()
        # r = repr(d)
        # assert r.startswith('<' + d.__class__.__name__)
        # todo

    @staticmethod
    def make_random(size: Optional[int] = None) -> Tuple[SimpleDataset,
                                                         np.ndarray,
                                                         np.ndarray]:
        """Create a random SimpleDataset object.

        Args:
            size: The desired number of rows in the random dataset.

        Returns:
            The random SimpleDataset object.

        """
        if size is None:
            size = 100
        inputs = np.random.rand(size, 5)
        targets = np.random.rand(size, 5)
        dataset = SimpleDataset(inputs, targets)
        return dataset, inputs, targets


class TestDatasetWrapper:
    """Tests for the DatasetWrapper class."""

    def test_wrapped_dataset_locking(self) -> None:
        """Test the wrapped dataset locking behaviour."""
        # Locking on by default
        wrapper = self.make_random(10)
        with pytest.raises(AssertionError):
            wrapper.dataset = TestSimpleDataset.make_random()[0]
        # Locking off
        wrapper = self.make_random(10, lock_dataset=False)
        wrapper.dataset = TestSimpleDataset.make_random()[0]

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
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

        return Wrapper(TestSimpleDataset.make_random(size)[0], *args, **kwargs)


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
            TestSimpleDataset.make_random(size)[0],
            partitions
        )


class TestConcatenatedDataset:
    """Tests for the ConcatenatedDataset class."""

    def test_indexing(self):
        concat, datasets = self.make_random()
        total_len = 0
        for i, d in enumerate(datasets):
            total_len += len(d)
            # Check rows match up
            # todo
            # for j, row in enumerate(d):
            #     assert dataset_rows_are_equal(row, concat[total_len + j - 1])
        # Overall length match
        assert len(concat) == total_len

    def test_repr(self) -> None:
        """Test the canonical string representation."""
        # todo
        pass

    @staticmethod
    def make_random() -> Tuple[ConcatenatedDataset, List[SimpleDataset]]:
        """Create a random AugmentedDataset object.

        Returns:
            The random ConcatenatedDataset, and the datasets concatenated.

        """
        datasets = [TestSimpleDataset.make_random(100)[0] for _ in range(3)]
        concat = ConcatenatedDataset(datasets)
        return concat, datasets


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
            -> Tuple[ResampledDataset, SimpleDataset]:
        """Create a random ResampledDataset object.

        Args:
            size:    The desired number of rows in the random dataset.
            samples: The desired samples to use in the object.

        Returns:
            The random ResampledDataset.

        """
        original = TestSimpleDataset.make_random(size)[0]
        resampled = ResampledDataset(original, samples)
        return resampled, original
