# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from ..datasets import BinnedDataset, TransformedDataset, ResampledDataset
from ..samplers import RandomTargetedResampler
from ..binners import ZeroNonZeroBinner, EqualBinner
from .classifiers import FeedForwardClassifier
from .misc import TrainableModelMixin, train_nn, FeedForwardBase


class FeedForwardRegressor(FeedForwardBase):
    """Simple multi-layer feed-forward network."""

    def __init__(self, *args, **kwargs) -> None:
        """Create a FeedForwardRegressor object."""
        super().__init__(*args, **kwargs)

    def forward(self, inputs):
        """Calculate the output of the model for the given inputs.

        Args:
            inputs: The inputs.

        Returns:
            The outputs.

        """
        return self.seq(inputs)


class HurdleRegressor(TrainableModelMixin):
    """Hurdle model.

    Used for two-'class' imbalanced regression problems.  E.g. take a
    zero-inflated Gaussian distributed target in a regression task.  The hurdle
    model first deals with the classification task of determining whether the
    target is zero or non-zero (the 'hurdle').  If non-zero, a separate
    regressor is used to predict the numerical value from the original
    input features at that idx.
    """

    def __init__(self, classifier=None, regressor=None) -> None:
        """Create a HurdleModel object.

        Args:
            classifier: The classifier for the first stage.
            regressor:  The regressor for the second stage.

        Returns:
            None

        """
        if classifier is None:
            classifier = FeedForwardClassifier()
        self.classifier = classifier
        if regressor is None:
            regressor = FeedForwardRegressor()
        self.regressor = regressor

    def forward(self, inputs):
        """Calculate the output of the model for the given inputs.

        Args:
            inputs: The inputs.

        Returns:
            The outputs.

        """
        preds = torch.argmax(self.classifier(inputs), 1).float()
        mask = preds != 0
        preds.masked_scatter_(mask, self.regressor(inputs))
        return preds

    def train_all(self, train_dataset: Dataset, val_dataset: Dataset) -> None:
        """Train the model with the given data.

        Convenience method with reasonable defaults.  Components may be trained
        directly with a custom/external implementation, if desired.

        Args:
            train_dataset: The training dataset.
            val_dataset:   The validation dataset.

        Returns:
            None.

        """

        # Get classifier datasets (0 vs. 1)
        clf_train_dataset = BinnedDataset(train_dataset, ZeroNonZeroBinner())
        clf_val_dataset = BinnedDataset(val_dataset, ZeroNonZeroBinner())

        # Train classifier
        train_nn(self.classifier, clf_train_dataset, clf_val_dataset, is_classification=True)

        # Get regressor datasets
        # Essentially resample out the zero class
        pos_only_sampler = RandomTargetedResampler(0.0, 0.0)
        reg_train_dataset = ResampledDataset(train_dataset, pos_only_sampler)
        reg_val_dataset = ResampledDataset(val_dataset, pos_only_sampler)

        # Train regressor
        train_nn(self.regressor, reg_train_dataset, reg_val_dataset)


class ReconRegressor(TrainableModelMixin):
    """Reconstituted regressor model."""

    def __init__(self, classifier, regressor, binner=None):
        """Create a ReconRegressorModel.

        Args:
            classifier: The classifier.
            regressor:  The regressor
            binner:     The binner.

        Returns:
            None

        """
        if classifier is None:
            classifier = FeedForwardClassifier()
        self.classifier = classifier
        if regressor is None:
            regressor = FeedForwardRegressor()
        self.regressor = regressor
        if binner is None:
            binner = EqualBinner()
        self.binner = binner

    def forward(self, inputs):
        """Calculate the output of the model for the given inputs.

        Args:
            inputs: The inputs.

        Returns:
            The outputs.

        """
        return self.regressor(
            self.classifier(inputs)
        )

    def train_all(self, train_dataset: Dataset, val_dataset: Dataset) -> None:
        """Train the model with the given data.

        Convenience method with reasonable defaults.  Components may be trained
        directly with a custom/external implementation, if desired.

        Args:
            train_dataset: The training dataset.
            val_dataset:   The validation dataset.

        Returns:
            None.

        """
        # Get classification dataset
        clf_train_dataset = BinnedDataset(train_dataset, self.binner)
        clf_val_dataset = BinnedDataset(val_dataset, self.binner)
        # Train classifier
        train_nn(self.classifier, clf_train_dataset, clf_val_dataset, is_classification=True)
        # Get regression dataset
        reg_train_dataset = TransformedDataset(
            train_dataset,
            self.classifier.forward
        )
        reg_val_dataset = TransformedDataset(
            val_dataset,
            self.classifier.forward
        )
        # Train regressor
        train_nn(self.regressor, reg_train_dataset, reg_val_dataset)

