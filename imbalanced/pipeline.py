# -*- coding: utf-8 -*-

"""
This file contains the basic definition for an imbalanced data 'pipeline'.
"""

import torch
from torch.nn import Module
from typing import List, Union, Optional
from .datasets import Dataset
from .preprocessors import Preprocessor, RandomSubsampler
from .postprocessors import Postprocessor
from .learner import LearningAlgorithm


class Pipeline:
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

    def train(self, dataset: Dataset) -> None:
        """Train the pipeline, including the network and the post-processors.

        Args:
            dataset: The dataset to use for training.

        Returns:
            None

        """

        # Validation
        assert isinstance(dataset, Dataset),\
            'Dataset argument must be an instance of Dataset (or a subclass)'

        # Pre-process the data
        for preprocessor in self.preprocessors:
            dataset = preprocessor.process(dataset)

        # Train the net
        # todo

        # Train the post-processors
        for postprocessor in self.postprocessors:
            postprocessor.train(dataset, self.net(dataset))

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

    def __repr__(self) -> str:
        """Get the canonical string representation for an instance of the
        object.

        Returns:
            The canonical string representation.

        """
        return '<%s(preprocessors=%s, net=%s, learner=%s, postprocessors=%s)>' \
               % (self.__class__.__name__, repr(self.preprocessors),
                   repr(self.net), repr(self.learner),
                   repr(self.postprocessors))


class AutoPipeline(Pipeline):
    """Automatically configured imbalanced data pipeline.

    Chooses reasonable defaults, based on the dataset given.
    """

    def __init__(self, dataset: Dataset) -> None:
        # todo: add dataset analysis for choosing

        # Pre-processor(s)
        preprocessors = RandomSubsampler(rate=0.5)

        # Net
        in_dim = dataset[0][0].size()[0]
        out_dim = 1
        hidden_dim = 100
        net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )

        # Learning algorithm
        learner = LearningAlgorithm(
            torch.nn.MSELoss(),
            torch.optim.Adam(
                net.parameters(),
                lr=0.1
            ),
            10
        )

        # Post-processor(s)
        postprocessors = None

        super().__init__(preprocessors, net, learner, postprocessors)
