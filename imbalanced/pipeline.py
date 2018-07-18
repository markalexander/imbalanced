# -*- coding: utf-8 -*-

"""
This file contains the basic definition for an imbalanced data 'pipeline'.
"""

import torch
from torch.nn import Module
from typing import List, Union, Optional, Tuple, Any
from .datasets import Dataset
from .preprocessors import Preprocessor, RandomSubsampler
from .postprocessors import Postprocessor
from .learner import LearningAlgorithm
from .meta import CanonicalArgsMixin


class Pipeline(CanonicalArgsMixin):
    """Imbalanced data pipeline.

    Defines an approach to the imbalanced data predictive modeling task.
    Essentially consists of defined processors for each intervention stage:

        Pre-processors -> Net architecture -> Learning algo -> Post-processors
    """

    def __init__(self,
                 preprocessors: Optional[Union[List[Preprocessor],
                                               Preprocessor]],
                 net: Module,
                 learner: LearningAlgorithm,
                 postprocessors: Optional[Union[List[Postprocessor],
                                                Postprocessor]]) -> None:
        """Create a pipeline object.

        Args:
            preprocessors:  A (list of) pre-processor(s), which may be empty
                            or None for no pre-processing.
            net:            A neural network.
            learner:        A learning algorithm specification.
            postprocessors: A (list of) post-processor(s), which may be empty
                            or None for no post-processing.

        """
        # Init
        self._preprocessors = None
        self._net = None
        self._learner = None
        self._postprocessors = None
        # Set
        if preprocessors is None:
            preprocessors = []
        self.preprocessors = preprocessors
        self.net = net
        self.learner = learner
        if postprocessors is None:
            postprocessors = []
        self.postprocessors = postprocessors

    @property
    def preprocessors(self) -> List[Preprocessor]:
        """Get the current pre-processor(s).

        Returns:
            The list of pre-processors.

        """
        return self._preprocessors

    @preprocessors.setter
    def preprocessors(self,
                      preprocessors: Union[List[Preprocessor],
                                           Preprocessor]) -> None:
        """Set the pre-processors.

        Args:
            preprocessors: The preprocessor or list of preprocessors to set.

        """
        if not isinstance(preprocessors, list):
            preprocessors = [preprocessors]
        self._preprocessors = []
        for p in preprocessors:
            assert issubclass(type(p), Preprocessor)
            self._preprocessors.append(p)

    @property
    def net(self) -> Module:
        """Get the current network object.

        Returns:
            The current network object.

        """
        return self._net

    @net.setter
    def net(self, net: Module):
        """Set the network object.

        Args:
            net: A neural network.

        """
        assert isinstance(net, Module)
        self._net = net

    @property
    def learner(self) -> LearningAlgorithm:
        """Get the current learning algorithm.

        Returns:
            The current learning algorithm.

        """
        return self._learner

    @learner.setter
    def learner(self, learner: LearningAlgorithm):
        """Set the learning algorithm.

        Args:
            learner: The learning algorithm to be set.

        """
        assert isinstance(learner, LearningAlgorithm)
        self._learner = learner

    @property
    def postprocessors(self) -> List[Preprocessor]:
        """Get the current post-processors.

        Returns:
            The current post-processors.

        """
        return self._postprocessors

    @postprocessors.setter
    def postprocessors(self,
                       postprocessors: Union[List[Postprocessor],
                                             Postprocessor]) -> None:
        """Set the post-processors.

        Args:
            postprocessors: The post-processors to be set.

        """
        if not isinstance(postprocessors, list):
            postprocessors = [postprocessors]
        self._postprocessors = []
        for p in postprocessors:
            assert issubclass(type(p), Postprocessor)
            self._postprocessors.append(p)

    def init(self) -> None:
        """(Re)-initialize the components in the pipeline.

        Includes e.g. weight initialization of the net, resetting and trained
        pre- or post- processors.

        Returns:
            None

        """
        # Pre- and post-processors
        for p in self.postprocessors + self.postprocessors:
            init_fn = getattr(p, 'init', None)
            if callable(init_fn):
                init_fn()
        # Net
        self.net.apply(self.learner.initializer)

    def train(self, train_dataset: Dataset,
              val_dataset: Optional[Dataset] = None) -> None:
        """Train the pipeline, including the network and the post-processors.

        Args:
            train_dataset: The dataset to use for training.
            val_dataset:   The dataset to use for validation.

        Returns:
            None

        """

        # Validation
        assert isinstance(train_dataset, Dataset),\
            'Training dataset argument must be an instance of' \
            'Dataset (or a subclass)'
        if val_dataset is not None:
            assert isinstance(val_dataset, Dataset),\
                'Validation dataset argument must be an instance of' \
                'Dataset (or a subclass)'

        # Pre-process the data
        for preprocessor in self.preprocessors:
            dataset = preprocessor.process(train_dataset)

        # Train the net
        # todo

        # Train the post-processors
        for postprocessor in self.postprocessors:
            postprocessor.train(train_dataset, self.net(train_dataset))

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the end-to-end prediction(s) for the given input(s).

        Includes any post-processing steps.  For net output only, call
        self.net(input) instead.

        Args:
            inputs: The inputs to make predictions for.

        Returns:
            The tensor of predictions.

        """
        output = self.net(inputs)
        for postprocessor in self.postprocessors:
            output = postprocessor.process(output)
        return output

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
            ('preprocessors', self.preprocessors),
            ('net', self.net),
            ('learner', self.learner),
            ('postprocessors', self.postprocessors)
        ]


class AutoPipeline(Pipeline):
    """Automatically configured imbalanced data pipeline.

    Chooses reasonable defaults, based on the dataset given.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        super().__init__(self.choose_preprocessors(), self.choose_net(),
                         self.choose_learner(), self.choose_postprocessors())

    def choose_preprocessors(self) -> List[Preprocessor]:
        """Get (choose) a set of pre-processors.

        Returns:
            The list of pre-processors.

        """
        return [RandomSubsampler(rate=0.5)]

    def choose_net(self) -> Module:
        """Get (choose) a network.

        Returns:
            The network object.

        """
        in_dim = self.dataset[0][0].size()[0]
        out_dim = 1
        hidden_dim = 100  # todo: between input and output?
        return torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def choose_learner(self) -> LearningAlgorithm:
        """Get (choose) the learning algorithm.

        Returns:
            The learning algorithm.

        """
        return LearningAlgorithm(
            torch.nn.MSELoss(),
            torch.optim.Adam(
                self.net.parameters(),
                lr=0.1
            ),
            10
        )

    def choose_postprocessors(self) -> List[Preprocessor]:
        """Get (choose) a set of pre-processors.

        Returns:
            The list of pre-processors.

        """
        return []
