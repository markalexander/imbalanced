# -*- coding: utf-8 -*-

import pytest
import torch
import numpy as np
from torch.utils.data import TensorDataset
from ..binners import ZeroNonZeroBinner


class TestZeroPositiveBinner:
    """Tests for the ZeroNonZeroBinner class."""

    def test_gives_correct_bins(self) -> None:
        dataset = TensorDataset(
            torch.Tensor([0.0, 1.0, 100.0, 0.0, 200.0])
        )
        bins = ZeroNonZeroBinner().get_bins(dataset)
        assert 2 == len(bins)
        assert (0.0,) == bins[0]
        assert (0.0, np.inf) == bins[1]