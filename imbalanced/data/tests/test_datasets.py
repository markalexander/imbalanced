# -*- coding: utf-8 -*-

import pytest
import numpy as np
from ..datasets import SimpleDataset


class TestSimpleDataset:

    def test_indexes_numpy_ndarray(self):
        inputs = np.random.rand(3, 3)
        targets = np.random.rand(3, 3)
        d = SimpleDataset(inputs, targets)
        for i in range(len(inputs)):
            assert (d[i][0].numpy() == inputs[i]).all()
            assert (d[i][1].numpy() == targets[i]).all()
        assert len(d) == len(inputs) == len(targets)

    def test_indexes_list_of_lists(self):
        inputs = np.random.rand(3, 3).tolist()
        targets = np.random.rand(3, 3).tolist()
        d = SimpleDataset(inputs, targets)
        for i in range(len(inputs)):
            assert (d[i][0].numpy() == np.array(inputs[i])).all()
            assert (d[i][1].numpy() == np.array(targets[i])).all()
        assert len(d) == len(inputs) == len(targets)

    def test_rejects_length_mismatch(self):
        with pytest.raises(AssertionError):
            inputs = np.random.rand(2, 3).tolist()
            targets = np.random.rand(3, 3).tolist()
            SimpleDataset(inputs, targets)
