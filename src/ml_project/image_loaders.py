"""Image loading utilities."""

import abc

import numpy as np
from PIL import Image


class AbstractImageLoader(metaclass=abc.ABCMeta):
    """
    Base class for image loaders.

    FIXME: see this thread in SO: https://stackoverflow.com/questions/36597318
    """

    @abc.abstractmethod
    def load(self, path):
        """Read an image file as an Image object."""
        raise NotImplementedError("load must be defined in derived classes")

    @abc.abstractmethod
    def load_np(self, path):
        """Read an image file and convert it to a numpy array."""
        raise NotImplementedError("load_np must be defined in derived classes")


class PillowLoader(AbstractImageLoader):
    """Image loader using pillow as backend."""

    def load(self, path):
        """Read an image as PIL.Image object."""
        sample = Image.open(path)
        if sample.mode != "RGB":
            sample = sample.convert("RGB")
        return sample

    def load_np(self, path):
        """Read an image with pillow and return a numpy array of its data."""
        return np.array(self.load(path))
