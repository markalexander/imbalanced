# -*- coding: utf-8 -*-

import pytest
from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _Loss
from ..learner import LearningAlgorithm


class DummyOptimizer(Optimizer):
    def __init__(self) -> None:
        pass


class DummyCriterion(_Loss):
    def __init__(self) -> None:
        pass


class TestLearningAlgorithm:

    def test_rejects_invalid_elements(self) -> None:
        criterion = DummyCriterion()
        optimizer = DummyOptimizer()
        with pytest.raises(AssertionError):
            LearningAlgorithm(1, optimizer)
        with pytest.raises(AssertionError):
            LearningAlgorithm(criterion, 1)
