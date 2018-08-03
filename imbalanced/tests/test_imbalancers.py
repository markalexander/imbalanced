# -*- coding: utf-8 -*-

import torch
from torch.utils.data import TensorDataset
from ..imblancers import InflatedFeatureFilteredDataset


original_inputs = torch.Tensor([
    [1, 1, 3, 4],
    [1, 1, 8, 7],
    [3, 2, 1, 0],
    [9, 1, 1, 1],
    [3, 2, 1, 4],
    [9, 2, 9, 3]
])

original_targets = torch.Tensor([
    1,
    2,
    3,
    4,
    5,
    6
])


class TestInflatedFeatureFilteredDataset:
    """Tests for the InflatedFeatureFilteredDataset class."""

    def test_match_al_target_is_changedl(self) -> None:
        """Tests whether the target is changed for an 'all' feature match."""
        dataset = InflatedFeatureFilteredDataset(
            TensorDataset(original_inputs, original_targets),
            {1: 2, 2: 1},
            match_type=all
        )
        # Check those that do pass filter keep original values
        for idx in [2, 4]:
            assert original_targets[idx] == dataset[idx][1]
        # Check those that don't pass filter are zeroed out
        for idx in [0, 1, 3, 5]:
            assert 0 == dataset[idx][1]

    def test_match_or_target_is_changed(self) -> None:
        """Tests whether the target is changed for an 'any' feature match."""
        dataset = InflatedFeatureFilteredDataset(
            TensorDataset(original_inputs, original_targets),
            {1: 2, 2: 1},
            match_type=any
        )
        # Check those that do pass filter keep original values
        for idx in [2, 3, 4, 5]:
            assert original_targets[idx] == dataset[idx][1]
        # Check those that don't pass filter are zeroed out
        for idx in [0, 1]:
            assert 0 == dataset[idx][1]

    def test_filter_features_are_removed(self) -> None:
        """Tests whether the features used for filtering are then removed
        from the features given for each row."""
        dataset = InflatedFeatureFilteredDataset(
            TensorDataset(original_inputs, original_targets),
            {1: 2}
        )
        for idx in range(len(dataset)):
            inputs, target = dataset[idx]
            assert original_inputs[idx][0] == inputs[0]
            assert original_inputs[idx][2] == inputs[1]
            assert original_inputs[idx][3] == inputs[2]

    def test_additional_features_are_removed(self) -> None:
        """Tests whether the given co-determined features are also removed."""
        dataset = InflatedFeatureFilteredDataset(
            TensorDataset(original_inputs, original_targets),
            {1: 2},
            codet_features=[2]
        )
        for idx in range(len(dataset)):
            inputs, target = dataset[idx]
            assert original_inputs[idx][0] == inputs[0]
            assert original_inputs[idx][3] == inputs[1]
