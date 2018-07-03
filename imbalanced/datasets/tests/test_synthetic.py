# -*- coding: utf-8 -*-

from ..synthetic import SKLearnSyntheticClassification,\
    SKLearnSyntheticRegression


class TestSKLearnSyntheticClassification:
    """Tests for the SKLearnSyntheticClassification class."""

    def test_general(self) -> None:
        dataset = SKLearnSyntheticClassification(n_samples=20)
        assert len(dataset) == 20


class TestSKLearnSyntheticRegression:
    """Tests for the SKLearnSyntheticRegression class."""

    def test_general(self) -> None:
        dataset = SKLearnSyntheticRegression(n_samples=20)
        assert len(dataset) == 20
