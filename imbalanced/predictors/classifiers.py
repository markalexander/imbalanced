# -*- coding: utf-8 -*-

import torch.nn.functional as F
from .misc import FeedForwardBase


class FeedForwardClassifier(FeedForwardBase):
    """Simple multi-layer feed-forward network."""

    def __init__(self, *args, **kwargs) -> None:
        """Create a FeedForwardRegressor object."""
        super().__init__(*args, **kwargs)

    def forward(self, inputs):
        """Calculate the output of the model for the given inputs.

        Args:
            inputs: The inputs.

        Returns:
            The outputs.

        """
        return F.log_softmax(self.seq(inputs))
