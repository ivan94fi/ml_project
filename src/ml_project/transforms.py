"""Transformations applied to samples."""

import copy

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose

from ml_project import utils


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
    std: float or tuple of float
        The standard deviation of the noise to apply. If this parameter is a
        tuple, it is assumed to be the range from which the actual standard
        deviation will be sampled randomly (uniformly) for each example

    """

    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std
        self.std_is_range = False
        try:
            self.std = float(self.std)
        except TypeError:
            if isinstance(self.std, (tuple, list)):
                if len(self.std) != 2:
                    raise ValueError("std must be a tuple with size 2")
                try:
                    self.std = tuple(map(float, self.std))
                    self.std_is_range = True
                except ValueError as e:
                    raise ValueError("std must be a tuple of floats") from e
            else:
                raise ValueError("std must be a float or tuple of floats")

    def __call__(self, sample):
        """Add gaussian noise to the input sample."""
        if self.std_is_range:
            std = utils.uniform(self.std[0], self.std[1])
        else:
            std = self.std
        noise = torch.randn(sample.shape) * std + self.mean
        return sample + noise

    def __repr__(self):
        """Print an adequate representation of the class."""
        return self.__class__.__name__ + "(mean={}, std={})".format(self.mean, self.std)


class ComposeCopies(Compose):
    """Compose several transforms together, by copying them.

    This class behaves exactly as torchvision.transforms.Compose, but the
    input transforms are deep-copied on initialization, with `copy.deepcopy` method.

    This class also implements the iterable interface as a convenience.

    Parameters
    ----------
    transforms : list
        The transforms to compose

    """

    def __init__(self, transforms):
        super().__init__(None)

        try:
            iter(transforms)
        except TypeError:
            raise ValueError("transforms parameter should be iterable") from None

        self.transforms = [
            copy.deepcopy(t) for t in _flatten_composed_transforms(transforms)
        ]

    def __getitem__(self, index):
        """Return the `index`-th transform."""
        return self.transforms[index]

    def __len__(self):
        """Return the number of transforms embedded in this composition."""
        return len(self.transforms)


def _flatten_composed_transforms(transforms):
    """Convert nested ComposeCopies objects to a single list of transforms."""
    flattened = []
    for transform in transforms:
        if isinstance(transform, ComposeCopies):
            flattened.extend(_flatten_composed_transforms(transform))
        else:
            flattened.append(transform)
    return flattened
