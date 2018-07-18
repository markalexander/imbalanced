# -*- coding: utf-8 -*-

from ..preprocessors import RandomSubsampler
from ...datasets.tests.test_generic import make_random_dataset


class TestRandomSubsampler:
    """Tests for the RandomSubsampler class."""

    def test_general(self) -> None:
        dataset = make_random_dataset(100)[0]
        resampler = RandomSubsampler(rate=0.5)
        dataset = resampler.resample(dataset)
        assert len(dataset) == 50
