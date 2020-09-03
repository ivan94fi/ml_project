"""Transformations applied to samples."""

import torch
import torchvision.transforms.functional as TF


class ResizeIfTooSmall:
    """
    Custom transformation implementing resize logic for small images.

    Parameters
    ----------
    size : int
        The minimum size that images should have.
    stretch : boolean
        If True, distort the image while resizing. If False, maintain
        the original aspect ratio (the default is True).

    """

    def __init__(self, size, stretch=True):
        self.size = size
        self.stretch = stretch

    def __call__(self, sample):
        """Apply the resize transformation."""
        width, height = sample.size
        if width < self.size or height < self.size:
            if not self.stretch:
                return TF.resize(sample, self.size)
            return sample.resize((self.size, self.size))
        return sample

    def __repr__(self):
        """Print an adequate representation of the class."""
        return self.__class__.__name__ + "(size={}, stretch={})".format(
            self.size, self.stretch
        )


class GaussianNoise:
    """
    Custom transform to add gaussian noise to a tensor.

    Parameters
    ----------
    mean: float
        The mean of the noise to apply
    std: float
        The standard deviation of the noise to apply

    """

    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """Add gaussian noise to the input sample."""
        noise = torch.randn(sample.shape) * self.std + self.mean
        return sample + noise

    def __repr__(self):
        """Print an adequate representation of the class."""
        return self.__class__.__name__ + "(mean={}, std={})".format(self.mean, self.std)
