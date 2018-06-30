# -*- coding: utf-8 -*-

import pytest
import numpy as np
from ..datasets import SimpleDataset, MaskedDataset


def dataset_rows_equal(row1, row2):
    """Utility functionf or comparing rows of form (inputs, outputs),
    as returned by __getitem__ on our datasets."""
    return (row1[0] == row2[0]).all() and (row1[1] == row2[1]).all()


class TestSimpleDataset:

    def test_indexes_numpy_ndarray(self):
        """Test whether the constructor accepts and correctly indexes numpy
        ndarray objects.
        """
        d, inputs, targets = self.make_random()
        for i in range(len(inputs)):
            assert (d[i][0].numpy() == inputs[i]).all()
            assert (d[i][1].numpy() == targets[i]).all()
        assert len(d) == len(inputs) == len(targets)

    def test_indexes_list_of_lists(self):
        """Test whether the constructor accepts and correctly indexes
        lists-of-lists.
        """
        _, inputs, targets = self.make_random()
        d = SimpleDataset(inputs.tolist(), targets.tolist())
        for i in range(len(inputs)):
            assert (d[i][0].numpy() == np.array(inputs[i])).all()
            assert (d[i][1].numpy() == np.array(targets[i])).all()
        assert len(d) == len(inputs) == len(targets)

    def test_rejects_length_mismatch(self):
        """Test whether the constructor raises an error for inputs and targets
        with differing lengths.
        """
        with pytest.raises(AssertionError):
            self.make_random((2, 3), (3, 3))

    @staticmethod
    def make_random(inputs_shape=(10, 5), targets_shape=(10, 5)):
        """Create a randomly-valued SimpleDataset object.

        :param inputs_shape:   desired shape for `inputs` tensor
        :param targets_shape:  desired shape for `targets` tensor
        :return:               dataset object, inputs, targets
        """
        inputs = np.random.rand(*inputs_shape)
        targets = np.random.rand(*targets_shape)
        d = SimpleDataset(inputs, targets)
        return d, inputs, targets
