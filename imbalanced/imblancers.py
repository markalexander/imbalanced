# -*- coding: utf-8 -*-

"""
This file contains definitions for the various imbalancer classes.

Essentially, an imbalancer wraps an existing dataset and makes it (more)
imbalanced, through a variety of methods.  The new object can then be used as a
dataset in its own right.
"""

from typing import Tuple, Any, Dict, List, Optional
import torch
from torch.utils.data import Dataset
from imbalanced.datasets import DatasetWrapper


class InflatedFeatureFilteredDataset(DatasetWrapper):
    """Wrapper which inflates a given dataset to have an increased number of
    target values of a particular value, e.g. zero.  Selects which features
    to set to this target value based on the values of input features.

    For example, in a house price prediction dataset we might want to set every
    row whose 'location' feature is not within a particular region, say 'spain'.
    All rows whose location feature is not spain would then be set to zero, and
    importantly this feature column would be removed in output.
    """

    def __init__(self, dataset: Dataset, filters: Dict[int, Any],
                 inflated_target: Any = 0.0,
                 codet_features: Optional[List[int]] = None,
                 match_type: Any = all) -> None:
        """Create an InflatedFeatureFilteredDataset object.

        Args:
            dataset:         The dataset to be inflated.
            filters:         The filters by which the data are inflated.  As a
                             dictionary specifying {feature_idx: filter_value}.
                             Then, any row that matches on all features
                             specified will retain its normal value.  All other
                             rows will have their target values replaced with
                             the inflated target value given.  Note that any
                             feature specified here will also be removed.
            inflated_target: The new target value to give those rows that don't
                             pass the filter.
            codet_features:  Other co-determined features to remove when
                             returning rows. For when we filter on some feature
                             that is also determined by other features that we
                             are not filtering by, and we would like to remove
                             these features too.  E.g. in a one-hot
                             representation.
            match_type:      Whether passing the filter should mean matching any
                             feature value or all feature values.  Pass one of
                             the Python built-ins `any` or `all`.

        Returns:
            None

        """
        super().__init__(dataset)
        self.filters = filters
        self.inflated_target = inflated_target
        if codet_features is None:
            codet_features = []
        self.match_type = match_type
        # Determine which features to keep in the rows
        all_features = set(range(len(self.dataset[0][0])))
        removed_features = set(
            codet_features + list(self.filters.keys()))
        self.keep_features = torch.LongTensor(
            list(all_features - removed_features))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a data row by index.

        Args:
            idx: The index of the desired row.

        Returns:
            The desired row (input, target)

        """
        features, target = self.dataset[idx]
        # Get target value
        passes_filter = self.match_type([v == features[k]
                                         for k, v in self.filters.items()])
        if not passes_filter:
            target = torch.Tensor([self.inflated_target])
        # Return with correct features removed
        return features.index_select(0, self.keep_features), target

    def __len__(self) -> int:
        """Get the total number of rows in the dataset.

        Returns:
            The number of rows.

        """
        return len(self.dataset)
