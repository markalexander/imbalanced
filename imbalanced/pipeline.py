# -*- coding: utf-8 -*-

"""
This file contains the basic definition for an imbalanced data 'pipeline'.
"""

from abc import ABC, abstractmethod
import torch
from torch.nn import Module
from torch.autograd import Variable
from typing import List, Union, Optional, Tuple, Any
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from .samplers import RandomClassResampler
from .learner import LearningAlgorithm
from .meta import CanonicalArgsMixin


class Pipeline(CanonicalArgsMixin):
    """Imbalanced data pipeline.

    Defines an approach to the imbalanced data predictive modeling task.
    Essentially consists of defined processors for each intervention stage:

        Pre-processors -> Net architecture -> Learning algo
    """

    def __init__(self,
                 samplers: Optional[Union[List[Sampler], Sampler]],
                 net: Module,
                 learner: LearningAlgorithm) -> None:
        """Create a pipeline object.

        Args:
            samplers:  A (list of) re-sampler(s), which may be empty or None for
                       no resampling.
            net:            A neural network.
            learner:        A learning algorithm specification.

        """
        # Init
        self._samplers = None
        self._net = None
        self._learner = None
        self._postprocessors = None
        # Set
        if samplers is None:
            samplers = []
        self.samplers = samplers
        self.net = net
        self.learner = learner

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

    def init(self) -> None:
        """(Re)-initialize the components in the pipeline.

        Includes e.g. weight initialization of the net, resetting and trained
        pre- or post- processors.

        Returns:
            None

        """
        self.net.apply(self.learner.initializer)

    def train(
            self,
            train_dataset: Dataset,
            val_dataset: Optional[Dataset] = None,
            logger: Optional['PipelineTrainingLogger'] = None
    ) -> None:
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
        if logger is not None:
            assert isinstance(logger, PipelineTrainingLogger),\
                'Logger must implement PipelineTrainingLogger'
        else:
            logger = NullPipelineTrainingLogger()

        # Pre-process the data
        # for preprocessor in self.samplers:
        #     dataset = preprocessor.process(train_dataset)

        # Dataloaders
        params = {'batch_size': 100, 'shuffle': True, 'num_workers': 4}
        train_dataloader = DataLoader(train_dataset, **params)
        val_dataloader = DataLoader(val_dataset, **params)

        # Train the net
        for epoch in range(10):

            # Loop through training batches
            for inputs, targets in train_dataloader:

                # Reset gradients
                self.net.zero_grad()

                # Variables
                inputs = Variable(inputs)
                targets = Variable(targets)

                # Forward pass
                outputs = self.net(inputs)
                loss = self.learner.criterion(outputs, targets)

                # Backward pass/update
                loss.backward()
                self.learner.optimizer.step()

                # Trigger end of batch event
                logger.on_net_training_batch_complete(self)

            # Trigger end of epoch
            logger.on_net_training_epoch_complete(self)

            # print(
            #     'Epoch: {} ## Train MSE: {}, Val MSE:'.format(
            #         epoch + 1,
            #         float(loss.data),
            #         # float(val_crit)
            #     )
            # )

        # Trigger end of net training
        logger.on_net_training_complete(self)

        # Train the post-processors
        # for postprocessor in self.postprocessors:
        #     postprocessor.train(train_dataset, self.net(train_dataset))

        # Trigger end of pipeline training
        logger.on_pipeline_training_complete(self)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the end-to-end prediction(s) for the given input(s).

        Includes any post-processing steps.  For net output only, call
        self.net(input) instead.

        Args:
            inputs: The inputs to make predictions for.

        Returns:
            The tensor of predictions.

        """
        return self.net(inputs)

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
            ('net', self.net),
            ('learner', self.learner)
        ]


class AutoPipeline(Pipeline):
    """Automatically configured imbalanced data pipeline.

    Chooses reasonable defaults, based on the dataset given.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        super().__init__(self.choose_samplers(dataset), self.choose_net(),
                         self.choose_learner())

    def choose_samplers(self, dataset) -> List[Sampler]:
        """Get (choose) a set of pre-processors.

        Returns:
            The list of pre-processors.

        """
        return [RandomClassResampler(dataset, 0, 0.5)]

    def choose_net(self) -> Module:
        """Get (choose) a network.

        Returns:
            The network object.

        """
        in_dim = self.dataset[0][0].size()[0]
        out_dim = 1
        hidden_dim = int(round((in_dim + out_dim) / 2))
        return torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
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


class PipelineTrainingLogger(ABC):
    """Interface for training session logger objects.

    Defines key triggers for events during the training process.
    """

    @abstractmethod
    def on_net_training_batch_complete(self, pipeline, pre_calc=None):
        pass

    @abstractmethod
    def on_net_training_epoch_complete(self, pipeline, pre_calc=None):
        pass

    @abstractmethod
    def on_net_training_complete(self, pipeline, pre_calc=None):
        pass

    @abstractmethod
    def on_pipeline_training_complete(self, pipeline, pre_calc=None):
        pass


class NullPipelineTrainingLogger(PipelineTrainingLogger):
    """Default 'null' logger for pipeline training.

    Discards all events.  Easier than doing many if-blocks in the training
    procedure.
    """

    def on_net_training_batch_complete(self, pipeline, pre_calc=None):
        pass

    def on_net_training_epoch_complete(self, pipeline, pre_calc=None):
        pass

    def on_net_training_complete(self, pipeline, pre_calc=None):
        pass

    def on_pipeline_training_complete(self, pipeline, pre_calc=None):
        pass
