"""Transformations applied to samples."""

import copy
import random

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose

from ml_project.utils import get_gaussian_kernel, transpose


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


class WhiteGaussianNoise:
    """
    Custom transform to add white gaussian noise to a tensor.

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
        except TypeError as t_e:
            if isinstance(self.std, (tuple, list)):
                if len(self.std) != 2:
                    raise ValueError("std must be a tuple with size 2") from t_e
                try:
                    self.std = tuple(map(float, self.std))
                    self.std_is_range = True
                except ValueError as e:
                    raise ValueError("std must be a tuple of floats") from e
            else:
                raise ValueError("std must be a float or tuple of floats") from t_e

    def __call__(self, sample):
        """Add gaussian noise to the input sample."""
        noise = self.generate_gaussian_noise(sample.shape, self.get_std())
        return sample + noise

    def generate_gaussian_noise(self, shape, std):
        """Get a noise tensor with values obtained from the gaussian distribution."""
        noise = torch.randn(shape) * std + self.mean
        return noise

    def get_std(self):
        """Sample a random std or return it if it is a number."""
        if self.std_is_range:
            return random.uniform(self.std[0], self.std[1])
        return self.std

    def __repr__(self):
        """Print an adequate representation of the class."""
        return self.__class__.__name__ + "(mean={}, std={})".format(self.mean, self.std)


class BrownGaussianNoise(WhiteGaussianNoise):
    """
    Custom transform to add brown gaussian noise to a tensor.

    First, a white gaussian noise tensor is generated, then this noise is
    filtered with a gaussian filter of the specified kernel_size.

    Parameters
    ----------
    mean: float
        The mean of the noise to apply
    std: float or tuple of float
        The standard deviation of the noise to apply. If this parameter is a
        tuple, it is assumed to be the range from which the actual standard
        deviation will be sampled randomly (uniformly) for each example
    kernel_std: float
        The gaussian filtern standard deviation
    kernel_size: int
        The gaussian filter width (if None, the size will be computed
        automatically based on kernel_std)

    """

    def __init__(self, kernel_std, kernel_size=None, mean=0.0, std=1.0):
        super().__init__(mean, std)
        self.kernel_std = kernel_std
        self.kernel_size = kernel_size
        self.kernel = get_gaussian_kernel(self.kernel_std, size=self.kernel_size)
        self.kernel_radius = self.kernel.shape[0] // 2
        self.kernel = self.kernel.expand(3, 1, 1, -1)
        self.pad_amount = (self.kernel_radius,) * 4

    def __call__(self, sample):
        """Add brown gaussian noise to the input sample."""
        std = self.get_std()
        white_noise = self.generate_gaussian_noise(sample.shape, std)
        brown_noise = self.filter_noise(white_noise)
        brown_noise = (brown_noise / brown_noise.std()) * std
        return sample + brown_noise

    def filter_noise(self, noise):
        """Filter the noise in input."""

        noise = noise.unsqueeze(0)
        row_filtered_noise = F.conv1d(
            F.pad(noise, self.pad_amount, mode="reflect"), self.kernel, groups=3
        )
        filtered_noise = F.conv1d(transpose(row_filtered_noise), self.kernel, groups=3)
        filtered_noise = transpose(filtered_noise.squeeze(0))

        return filtered_noise

    def __repr__(self):
        """Print an adequate representation of the class."""
        return self.__class__.__name__ + "(kernel_std={}, mean={}, std={})".format(
            self.kernel_std, self.mean, self.std
        )


class PoissonNoise:
    """
    Custom transform to add Poisson noise to a tensor.

    Parameters
    ----------
    lmbda: float or tuple of float
        The standard deviation of the noise to apply. If this parameter is a
        tuple, it is assumed to be the range from which the actual standard
        deviation will be sampled randomly (uniformly) for each example

    """

    def __init__(self, lmbda=1.0):
        self.lmbda = lmbda
        self.lmbda_is_range = False
        try:
            self.lmbda = float(self.lmbda)
        except TypeError as t_e:
            if isinstance(self.lmbda, (tuple, list)):
                if len(self.lmbda) != 2:
                    raise ValueError("lmbda must be a tuple with size 2") from t_e
                try:
                    self.lmbda = tuple(map(float, self.lmbda))
                    self.lmbda_is_range = True
                except ValueError as e:
                    raise ValueError("lmbda must be a tuple of floats") from e
            else:
                raise ValueError("lmbda must be a float or tuple of floats") from t_e

    def __call__(self, sample):
        """Add Poisson noise to the input sample."""
        lmbda = self.get_lmbda()
        return torch.poisson(lmbda * (sample + 0.5)) / lmbda - 0.5

    def get_lmbda(self):
        """Sample a random lmbda or return it if it is a number."""
        if self.lmbda_is_range:
            return random.uniform(self.lmbda[0], self.lmbda[1])
        return self.lmbda

    def __repr__(self):
        """Print an adequate representation of the class."""
        return self.__class__.__name__ + "(lmbda={})".format(self.lmbda)


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
