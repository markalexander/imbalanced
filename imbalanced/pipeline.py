# -*- coding: utf-8 -*-

"""
This file contains the basic definition for an imbalanced data 'pipeline'.
"""

from torch.nn import Module
from ..preprocessors import Preprocessor
from ..postprocessors import Postprocessor
from .learner import LearningAlgorithm


class Pipeline:
    """Imbalanced data pipeline.

    Defines an approach to the imbalanced data predictive modeling task.
    Essentially consists of defined processors for each intervention stage:

        Pre-processors -> Net architecture -> Learning algo -> Post-processors
    """

    def __init__(self, preprocessors, net, learner, postprocessors):
        """Create a Pipeline object.

        :param preprocessors:   a (list of) pre-processor(s), which may be empty
                                or None for no pre-processing
        :type  preprocessors:   Preprocessor or list[Preprocessor] or None
        :param net:             a neural network object
        :type  net:             torch.nn.Module
        :param learner:         a learning algorithm specification
        :type  learner:         LearningAlgorithm
        :param postprocessors:  a (list of) post-processor(s), which may be
                                empty or None for no post-processing
        :type  postprocessors:  Postprocessor or list[Postprocessor] or None
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
    def preprocessors(self):
        """Get the current pre-processor(s).

        :return:  the list of pre-processors
        :rtype:   list[Preprocessor]
        """
        return self._preprocessors

    @preprocessors.setter
    def preprocessors(self, preprocessors):
        """Set the pre-processors.

        :param preprocessors:  the preprocessor or list of preprocessors to set
        :type  preprocessors:  Preprocessor or list[Preprocessor]
        :return:               None
        :rtype:                None
        """
        if not isinstance(preprocessors, list):
            preprocessors = [preprocessors]
        for p in preprocessors:
            assert issubclass(type(p), Preprocessor)
            self._preprocessors.append(p)

    @property
    def net(self):
        """Get the current network object.

        :return:
        """
        return self._net

    @net.setter
    def net(self, net):
        """Set the network object.

        :param net:
        :return:
        """
        assert isinstance(net, Module)
        self._net = net

    @property
    def learner(self):
        """Get the current learning algorithm.

        :return:
        """
        return self._learner

    @learner.setter
    def learner(self, learner):
        """Set the learning algorithm.

        :param learner:
        :return:
        """
        assert isinstance(learner, LearningAlgorithm)
        self._learner = learner

    @property
    def postprocessors(self):
        """Get the current post-processors.

        :return:
        """
        return self._postprocessors

    @postprocessors.setter
    def postprocessors(self, postprocessors):
        """Set the post-processors.

        :param postprocessors:
        :return:
        """
        if not isinstance(postprocessors, list):
            postprocessors = [postprocessors]
        for p in postprocessors:
            assert issubclass(type(p), Postprocessor)
            self._postprocessors.append(p)

    def train(self, dataset):
        """Train the pipeline, including the network and the post-processors.

        :return:
        """
        # Pre-process the data
        for preprocessor in self.preprocessors:
            dataset = preprocessor.process(dataset)

        # Train the net
        # todo

        # Train the post-processors
        for postprocessor in self.postprocessors:
            postprocessor.train(dataset, self.net(dataset))

    def predict(self, inputs):
        """Return the end-to-end prediction(s) for the given input(s).

        Includes any post-processing steps.  For net output only, call
        self.net(input) instead.

        :param inputs:
        :return:
        """
        output = self.net(inputs)
        for postprocessor in self.postprocessors:
            output = postprocessor.process(output)
        return output

    def __call__(self, inputs):
        """Shortcut for the predict() method.

         For consistency with PyTorch model behaviour.

        :param inputs:
        :return:
        """
        return self.predict(inputs)
