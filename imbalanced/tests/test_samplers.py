# -*- coding: utf-8 -*-

import pytest
import torch
from torch.utils.data import TensorDataset
from ..samplers import RandomResampler, RandomTargetedResampler


dataset = TensorDataset(
    torch.Tensor([
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4]
    ]),
    torch.Tensor([
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        1,
    ]),
)


class TestRandomResampler:
    """Tests for the RandomResampler class."""

    def test_sub_sampling(self) -> None:
        assert 5 == len(RandomResampler(dataset, 0.5).indices)
        assert 3 == len(RandomResampler(dataset, 0.3).indices)

    def test_super_sampling(self) -> None:
        assert 15 == len(RandomResampler(dataset, 1.5).indices)
        assert 23 == len(RandomResampler(dataset, 2.3).indices)

    def test_identity_sampling(self) -> None:
        assert 10 == len(RandomResampler(dataset, 1.0).indices)

    def test_rejects_negative_rate(self) -> None:
        with pytest.raises(AssertionError):
            RandomResampler(dataset, -0.1)


class TestTargetedClassResampler:
    """Tests for the RandomTargetedResampler class."""

    def test_sub_sampling(self) -> None:
        # 1/2 of class 0 (0.5 * 6 = 3) + all of class 1 (4) = 7
        assert 7 == len(RandomTargetedResampler(dataset, 0, 0.5).indices)
        # 1/3 of class 0 (0.3... * 6 = 2) + all of class 1 (4) = 6
        assert 6 == len(RandomTargetedResampler(dataset, 0, 1 / 3).indices)
        # all of class 0 (6) + 40% of class 1 (0.4 * 4 = 2) = 8
        assert 8 == len(RandomTargetedResampler(dataset, 1, 0.4).indices)

    def test_super_sampling(self) -> None:
        # all of class 0 (6) + 150% of class 1 (1.5 * 4 = 6) = 12
        assert 12 == len(RandomTargetedResampler(dataset, 1, 1.5).indices)
        # all of class 0 (6) + 120% of class 1 (1.2 * 4 = 5) = 11
        assert 11 == len(RandomTargetedResampler(dataset, 1, 1.2).indices)

    def test_identity_sampling(self) -> None:
        assert 10 == len(RandomTargetedResampler(dataset, 0, 1.0).indices)
        assert 10 == len(RandomTargetedResampler(dataset, 1, 1.0).indices)

    def test_rejects_negative_rate(self) -> None:
        with pytest.raises(AssertionError):
            RandomTargetedResampler(dataset, 0, -0.1)
