# -*- coding: utf-8 -*-

import pytest
from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _Loss
from ..learner import LearningAlgorithm


# noinspection PyAbstractClass,PyMissingConstructor
class DummyOptimizer(Optimizer):
    def __init__(self) -> None:
        pass


# noinspection PyAbstractClass,PyMissingConstructor
class DummyCriterion(_Loss):
    def __init__(self) -> None:
        pass


class TestLearningAlgorithm:

    def test_rejects_invalid_elements(self) -> None:
        criterion = DummyCriterion()
        optimizer = DummyOptimizer()
        with pytest.raises(AssertionError):
            # noinspection PyTypeChecker
            LearningAlgorithm(1, optimizer)
        with pytest.raises(AssertionError):
            # noinspection PyTypeChecker
            LearningAlgorithm(criterion, 1)
