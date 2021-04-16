"""Transformations applied to samples."""

import copy
import random
import string

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import Compose

from ml_project.utils import get_gaussian_kernel, transpose


def _float_or_float2tuple(param):
    is_range = False
    try:
        param = float(param)
    except TypeError as t_e:
        if isinstance(param, (tuple, list)):
            if len(param) != 2:
                raise ValueError("The parameter must be a tuple with size 2") from t_e
            try:
                param = tuple(map(float, param))
                is_range = True
            except ValueError as e:
                raise ValueError("The parameter must be a tuple of floats") from e
        else:
            raise ValueError(
                "The parameter must be a float or tuple of floats"
            ) from t_e
    return param, is_range


def no_op_transform(img):
    """No operation transform.

    Returns the input as is.
    """
    return img


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

    def __call__(self, sample):
        """Add gaussian noise to the input sample."""
        noise = self.generate_gaussian_noise(sample.shape, self.std)
        return sample + noise

    def generate_gaussian_noise(self, shape, std):
        """Get a noise tensor with values obtained from the gaussian distribution."""
        noise = torch.randn(shape) * std + self.mean
        return noise

    @property
    def std(self):
        """Sample a random std or return it if it is a number."""
        if self.is_range:
            return random.uniform(self._std[0], self._std[1])
        return self._std

    @std.setter
    def std(self, value):
        try:
            self._std, self.is_range = _float_or_float2tuple(value)
        except ValueError as e:
            raise ValueError("std malformed") from e

    def __repr__(self):
        """Print an adequate representation of the class."""
        return self.__class__.__name__ + "(mean={}, std={})".format(
            self.mean, self._std
        )


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
        std = self.std
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
            self.kernel_std, self.mean, self._std
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

    def __call__(self, sample):
        """Add Poisson noise to the input sample."""
        lmbda = self.lmbda
        return torch.poisson(lmbda * (sample + 0.5)) / lmbda - 0.5

    @property
    def lmbda(self):
        """Sample a random lmbda or return it if it is a number."""
        if self.is_range:
            return random.uniform(self._lmbda[0], self._lmbda[1])
        return self._lmbda

    @lmbda.setter
    def lmbda(self, value):
        try:
            self._lmbda, self.is_range = _float_or_float2tuple(value)
        except ValueError as e:
            raise ValueError("lmbda malformed") from e

    def __repr__(self):
        """Print an adequate representation of the class."""
        return self.__class__.__name__ + "(lmbda={})".format(self._lmbda)


def get_coverage(mask):
    """Calculate coverage as average value over a binary mask."""
    tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes()))
    return tensor.sum().item() / tensor.numel()


class TextualNoise:
    """
    Custom transform to add text noise to a tensor.

    Parameters
    ----------
    coverage: float or tuple of float
        A value (or range of values) in [0,1] that represents the percentage
        of pixels that should be covered by text in the noisy image
    font_filename: str
        The name of the font file to load. Defaults to DejaVu serif.
    sizes_range: iterable
        The font sizes to use when corrupting the image
    """

    def __init__(self, coverage=0.3, font_filename=None, sizes_range=None):
        self.coverage = coverage

        self.characters = string.ascii_letters + string.punctuation + string.digits

        if font_filename is None:
            font_filename = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
        self.font_filename = font_filename

        if sizes_range is None:
            sizes_range = range(10, 32 + 1, 2)
        self.sizes_range = sizes_range

        self.fonts = self._read_fonts()

    def _read_fonts(self):
        return [
            ImageFont.truetype(self.font_filename, size) for size in self.sizes_range
        ]

    def __call__(self, sample):
        """Add textual noise to the input sample."""
        self.add_text(sample, self.coverage)

        return sample

    def add_text(self, im, coverage):
        """Corrupt the image with the given coverage percentage."""
        draw = ImageDraw.Draw(im)
        mask = Image.new("L", im.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        while get_coverage(mask) < coverage:
            x = random.uniform(0, im.size[0])
            y = random.uniform(0, im.size[1])
            text = "".join(random.choices(self.characters, k=random.randint(15, 30)))
            font = random.choice(self.fonts)
            color = tuple(random.randint(0, 255) for _ in "RGB")
            draw.text((x, y), text, font=font, fill=color, anchor="mm")
            mask_draw.text((x, y), text, font=font, fill=1, anchor="mm")

        return im

    @property
    def coverage(self):
        """Sample a random coverage or return it if it is a number."""
        if self.is_range:
            return random.uniform(self._coverage[0], self._coverage[1])
        return self._coverage

    @coverage.setter
    def coverage(self, value):
        try:
            self._coverage, self.is_range = _float_or_float2tuple(value)
        except ValueError as e:
            raise ValueError("coverage malformed") from e
        if self.is_range:
            if not 0 <= self._coverage[0] <= 1 or not 0 <= self._coverage[1] <= 1:
                raise ValueError("coverage is a probability, must be in [0, 1]")
        else:
            if not 0 <= self._coverage <= 1:
                raise ValueError("coverage is a probability, must be in [0, 1]")

    def __repr__(self):
        """Print an adequate representation of the class."""
        return self.__class__.__name__ + "(coverage={})".format(self._coverage)

    def __getstate__(self):
        """Pickle."""
        state = self.__dict__.copy()
        # Remove the unpicklable fonts attribute
        del state["fonts"]
        return state

    def __setstate__(self, state):
        """Unpickle."""
        # Restore normal instance attributes
        self.__dict__.update(state)
        # Restore fonts from other attributes
        self.fonts = self._read_fonts()


class RandomImpulseNoise:
    """
    Custom transform to add random valued impulse noise.

    Pixels in input tensor are replaced with a random color with probability
    p and retain their color with probability 1-p. The random color is sampled
    uniformly from [0, 1]^3.

    Parameters
    ----------
    p: float or tuple of float
        A value (or range of values) in [0,1] that represents the probability
        that a pixel value is changed.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """Add random valued impulse noise to the input sample."""
        # Pixels values should be changed where mask is True
        mask = torch.rand(sample.shape[1], sample.shape[2])
        mask = mask.lt(self.p)

        noise = torch.rand_like(sample, dtype=torch.float) - 0.5

        # sample[i,j] = noise[i,j] if mask[i,j] else sample[i,j]
        sample = torch.where(mask, noise, sample)

        return sample

    @property
    def p(self):
        """Sample a random p or return it if it is a number."""
        if self.is_range:
            return random.uniform(self._p[0], self._p[1])
        return self._p

    @p.setter
    def p(self, value):
        try:
            self._p, self.is_range = _float_or_float2tuple(value)
        except ValueError as e:
            raise ValueError("p malformed") from e
        if self.is_range:
            if not 0 <= self._p[0] <= 1 or not 0 <= self._p[1] <= 1:
                raise ValueError("p is a probability, must be in [0, 1]")
        else:
            if not 0 <= self._p <= 1:
                raise ValueError("p is a probability, must be in [0, 1]")

    def __repr__(self):
        """Print an adequate representation of the class."""
        return self.__class__.__name__ + "(p={})".format(self._p)


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

    def __init__(self, *transforms):
        super().__init__(None)

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
        if isinstance(transform, (ComposeCopies, list, tuple)):
            flattened.extend(_flatten_composed_transforms(transform))
        else:
            flattened.append(transform)
    return flattened
