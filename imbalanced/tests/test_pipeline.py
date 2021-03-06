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


predictor = DummyNet()


class TestPipeline:

    def test_constructor_rejects_invalid_elements(self) -> None:
        pass
        # # Invalid predictor
        # with pytest.raises(AssertionError):
        #     # noinspection PyTypeChecker
        #     Pipeline(1, None)
        # Invalid sampler
        # with pytest.raises(AssertionError):
        #     # noinspection PyTypeChecker
        #     Pipeline(predictor, 1)
        # with pytest.raises(AssertionError):
        #     # noinspection PyTypeChecker
        #     Pipeline(None, None)

    def test_train_rejects_invalid_elements(self) -> None:
        pipeline = Pipeline(None, predictor)
        with pytest.raises(AssertionError):
            # noinspection PyTypeChecker
            pipeline.train_all(1, 1)
