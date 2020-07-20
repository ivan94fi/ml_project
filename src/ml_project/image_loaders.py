"""Image loading utilities."""

import abc

import numpy as np

# fmt: off
try:
    import accimage
except ImportError:
    print("accimage not found. Using pillow as image loading backend")
    accimage = None
    from PIL import Image
else:
    # accimage imported. Set it as backend in torchvision
    import torchvision
    torchvision.set_image_backend("accimage")
# fmt: on


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


class AccimageLoader(AbstractImageLoader):
    """Image loader using accimage as backend."""

    def load(self, path):
        """Read an image as accimage.Image object."""
        try:
            return accimage.Image(path)
        except OSError:
            return PillowLoader().load(path)

    def load_np(self, path):
        """Read an image with accimage and return a numpy array of its data."""
        return AccimageLoader.accimage_to_np(self.load(path))

    @staticmethod
    def accimage_to_np(image):
        """Convert the image to a numpy array with shape (width, height, channels)."""
        image_np = np.empty([image.channels, image.height, image.width], dtype=np.uint8)
        image.copyto(image_np)
        image_np = np.transpose(image_np, (1, 2, 0))
        return image_np
