# pylint: disable=R0903
"""Loss classes for the project."""
import warnings

import torch


class AnnealedL0Loss:
    """Annealed "L0" loss as described in Noise2Noise paper."""

    def __init__(self, total_epochs, initial_gamma=2, epsilon=1e-8):
        self.total_epochs = total_epochs
        if self.total_epochs == 1:
            warnings.warn(
                "The total epochs number is 1: "
                "using initial_gamma as the only gamma value."
            )
            self.total_epochs = 2
        self.initial_gamma = initial_gamma
        self.epsilon = epsilon

    def __call__(self, output, target, epoch):
        """Obtain the loss value from the network output and the target."""
        gamma = (
            self.initial_gamma * (self.total_epochs - epoch) / (self.total_epochs - 1)
        )
        result = (torch.abs(output - target) + self.epsilon) ** gamma
        result = torch.mean(result)
        return result
