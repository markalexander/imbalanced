# -*- coding: utf-8 -*-

import pytest
import torch.nn as nn
from torch.nn.modules.loss import MSELoss
from torch.optim.adam import Adam
from ..pipeline import Pipeline
from ..learner import LearningAlgorithm


# noinspection PyAbstractClass
class DummyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2)


net = DummyNet()
learner = LearningAlgorithm(MSELoss(), Adam(net.parameters()))


class TestPipeline:

    def test_constructor_rejects_invalid_elements(self) -> None:
        # Invalid pre-processor(s)
        with pytest.raises(AssertionError):
            # noinspection PyTypeChecker
            Pipeline(1, net, learner, None)
        # Invalid net
        with pytest.raises(AssertionError):
            # noinspection PyTypeChecker
            Pipeline(None, 1, learner, None)
        with pytest.raises(AssertionError):
            # noinspection PyTypeChecker
            Pipeline(None, None, learner, None)
        # Invalid learning algo
        with pytest.raises(AssertionError):
            # noinspection PyTypeChecker
            Pipeline(None, net, 1, None)
        with pytest.raises(AssertionError):
            # noinspection PyTypeChecker
            Pipeline(None, net, None, None)
        # Invalid post-processor(s)
        with pytest.raises(AssertionError):
            # noinspection PyTypeChecker
            Pipeline(None, net, learner, 1)

    def test_train_rejects_invalid_elements(self) -> None:
        pipeline = Pipeline(None, net, learner, None)
        with pytest.raises(AssertionError):
            # noinspection PyTypeChecker
            pipeline.train(1)
