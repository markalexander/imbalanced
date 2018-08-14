# -*- coding: utf-8 -*-

"""
This file contains the basic definition for an imbalanced data 'pipeline'.
"""

import torch
from typing import List, Union, Optional, Tuple, Any
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from .datasets import ResampledDataset
from .samplers import RandomClassResampler
from .meta import CanonicalArgsMixin


class Pipeline(CanonicalArgsMixin):
    """Imbalanced data pipeline."""

    def __init__(self, samplers: Optional[Union[List[Sampler], Sampler]],
                 predictor: Any) -> None:
        """Create a pipeline object.

        Args:
            samplers:  A (list of) re-sampler(s), which may be empty or None for
                       no resampling.
            predictor: The predictor.

        """
        self._samplers = None
        if samplers is None:
            samplers = []
        self.samplers = samplers
        self.predictor = predictor

    @property
    def samplers(self) -> List[Sampler]:
        """Get the current pre-processor(s).

        Returns:
            The list of pre-processors.

        """
        return self._samplers

    @samplers.setter
    def samplers(self, samplers: Union[List[Sampler], Sampler]) -> None:
        """Set the pre-processors.

        Args:
            samplers: The preprocessor or list of samplers to set.

        """
        if not isinstance(samplers, list):
            samplers = [samplers]
        self._samplers = []
        for s in samplers:
            assert issubclass(type(s), Sampler)
            self._samplers.append(s)

    def train(self, train_dataset: Dataset, val_dataset: Dataset) -> None:
        """Train the pipeline.
        
        Convenience method that assumes the predictor has a train_all() method.
        Otherwise, you should train the predictor externally.

        Args:
            dataset: The dataset to use for training.

        Returns:
            None

        """
        # Argument validation
        assert isinstance(train_dataset, Dataset),\
            'Training dataset argument must be an instance of' \
            'Dataset (or a subclass)'
        assert isinstance(val_dataset, Dataset),\
            'Validation dataset argument must be an instance of' \
            'Dataset (or a subclass)'
        # Resample the data
        for sampler in self.samplers:
            train_dataset = ResampledDataset(train_dataset, sampler)
            val_dataset = ResampledDataset(val_dataset, sampler)
        # Train the predictor
        self.predictor.train_all(train_dataset, val_dataset)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the end-to-end prediction(s) for the given input(s).

        Includes any post-processing steps.  For net output only, call
        self.predictor(input) instead.

        Args:
            inputs: The inputs to make predictions for.

        Returns:
            The tensor of predictions.

        """
        return self.predictor(inputs)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Shortcut for the predict() method.

         For consistency with PyTorch model API.

        Args:
            inputs: The inputs to make predictions for.

        Returns:
            The tensor of predictions.

        """
        return self.predict(inputs)

    @property
    def c_args(self) -> List[Tuple[str, Any]]:
        """Get the canonical (ordered) list of arguments ('c-args') which define
        the current object.

        Returns:
            The arguments, as a list of tuples (arg_name, arg_value).

        """
        return [
            ('samplers', self.samplers),
            ('predictor', self.predictor),
        ]


class AutoPipeline(Pipeline):
    """Automatically configured imbalanced data pipeline.

    Chooses reasonable defaults, based on the dataset given.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        super().__init__(self.choose_samplers(dataset),
                         self.choose_predictor(dataset))

    @staticmethod
    def choose_samplers(dataset) -> List[Sampler]:
        """Get (choose) a set of pre-processors.

        Returns:
            The list of pre-processors.

        """
        return [RandomClassResampler(dataset, 0, 0.5)]

    @staticmethod
    def choose_predictor(dataset) -> Any:
        """Get (choose) a network.

        Returns:
            The network object.

        """
        in_dim = dataset[0][0].size()[0]
        out_dim = 1
        hidden_dim = int(round((in_dim + out_dim) / 2))
        return torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )
