# -*- coding: utf-8 -*-

import pytest
import numpy as np
from ..datasets import SimpleDataset, MaskedDataset


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


class TestMaskedDataset:

    def test_masked_indexes_correctly(self):
        mask = np.array([True, False, True, False, True], dtype=np.bool)
        d_masked, d_orig = self.make_random(
            inputs_shape=(5, 5),
            targets_shape=(5, 5),
            mask=mask)
        assert len(d_masked) == 3
        i_masked = 0
        for i_orig in range(len(d_orig)):
            if mask[i_orig]:
                assert (d_orig[i_orig][0] == d_masked[i_masked][0]).all()
                assert (d_orig[i_orig][1] == d_masked[i_masked][1]).all()
                i_masked += 1

    def test_all_rows_active_by_default(self):
        d_masked, d_orig = self.make_random()
        assert len(d_masked) == len(d_orig)
        for i in range(len(d_orig)):
            assert (d_orig[i][0] == d_masked[i][0]).all()
            assert (d_orig[i][1] == d_masked[i][1]).all()

    def test_rejects_invalid_mask(self):
        # Not an ndarray
        with pytest.raises(AssertionError):
            self.make_random(mask=1)
        # Not the right length
        with pytest.raises(AssertionError):
            self.make_random(inputs_shape=(2, 5), targets_shape=(2, 5),
                             mask=np.array([True, False, True], dtype=np.bool))
        # Right length but not overall shape
        with pytest.raises(AssertionError):
            self.make_random(inputs_shape=(2, 5), targets_shape=(2, 5),
                             mask=np.array([[True, False], [True, False]],
                                           dtype=np.bool))

    @staticmethod
    def make_random(inputs_shape=(10, 5), targets_shape=(10, 5), mask=None):
        """Create a randomly MaskedDataset object.

        :param inputs_shape:   desired shape for `inputs` tensor
        :param targets_shape:  desired shape for `targets` tensor
        :param mask:           mask to be passed to the constructor
        :return:               masked dataset, original dataset
        """
        inputs = np.random.rand(*inputs_shape)
        targets = np.random.rand(*targets_shape)
        d = SimpleDataset(inputs, targets)
        m = MaskedDataset(d, mask)
        return m, d
