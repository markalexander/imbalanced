# -*- coding: utf-8 -*-

import pytest
import numpy as np
from ..preprocessors import Re
from ...data.tests.test_datasets import TestSimpleDataset, dataset_rows_equal


class TestSubsampledDataset:

    def test_masked_indexes_correctly(self):
        """Test whether the masked indexing/length works correctly."""
        mask = np.array([True, False, True, False, True], dtype=np.bool)
        d_masked, d_orig = self.make_random(
            inputs_shape=(5, 5),
            targets_shape=(5, 5),
            mask=mask)
        assert len(d_masked) == 3
        i_masked = 0
        for i_orig in range(len(d_orig)):
            if mask[i_orig]:
                assert dataset_rows_equal(d_orig[i_orig], d_masked[i_masked])
                i_masked += 1

    def test_all_rows_active_by_default(self):
        """Test whether the default behaviour (no masking) works correctly."""
        d_masked, d_orig = self.make_random()
        assert len(d_masked) == len(d_orig)
        for i in range(len(d_orig)):
            assert dataset_rows_equal(d_orig[i], d_masked[i])

    def test_rejects_invalid_mask(self):
        """Test whether the mask() method, and by extension the constructor,
        rejects invalid input."""
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
        d = TestSimpleDataset.make_random(inputs_shape, targets_shape)
        s = SubsampledDataset(d, mask)
        return s, d
