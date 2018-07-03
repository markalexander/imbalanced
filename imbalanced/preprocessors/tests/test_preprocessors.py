# -*- coding: utf-8 -*-

from ..preprocessors import RandomSubsampler
from ...datasets.tests.test_generic import TestSimpleDataset


class TestRandomSubsampler:
    """Tests for the RandomSubsampler class."""

    def test_general(self) -> None:
        dataset = TestSimpleDataset.make_random(100)[0]
        resampler = RandomSubsampler(rate=0.5)
        dataset = resampler.resample(dataset)
        assert len(dataset) == 50
