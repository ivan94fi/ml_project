"""Classes to load images from the used datasets."""

import os
import warnings
from collections import namedtuple

from torch.utils.data import Dataset, random_split

from ml_project.image_loaders import PillowLoader


def is_image_file(filename):
    """Check whether the given path refers to an image file using its exstension."""
    return os.path.splitext(filename)[1].lower() in [".jpg", ".jpeg", ".png"]


TrainingPair = namedtuple("TrainingPair", ["sample", "target"])


# See example of superresolution in torch examples
class BSDS300(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return 0


class ImageNet(Dataset):
    """Custom torch.utils.data.Dataset subclass for ImageNet validation split.

    Parameters
    ----------
    root_dir : str
        Path to dataset directory (where extracted jpg images reside)
    image_paths : list of str, optional
        If passed, use these filenames to contruct the dataset,
        do not parse the root directory (the default is None).
    transforms : callable, optional
        Functions or transforms to apply to data (the default is None).
    target_transforms : callable, optional
        Functions or transforms to apply to targets (the default is None).
    loader : AbstractImageLoader implementation, optional
        The image loader to use: defaults to PillowLoader

    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        root_dir,
        image_paths=None,
        transforms=None,
        target_transforms=None,
        loader=None,
    ):
        if not os.path.isdir(root_dir):
            raise ValueError("root_dir does not point to a valid directory")
        self.root_dir = root_dir

        if image_paths is None:
            self._construct_image_paths()
        else:
            self.image_paths = image_paths

        self.transforms = transforms
        self.target_transforms = target_transforms

        if loader is None:
            self.loader = PillowLoader()
        else:
            self.loader = loader

    def __getitem__(self, idx):
        """
        Return a training pair (sample, target).

        The image at the specified index is read from disk, processed with the
        predefined transforms and returned as a (sample, target) tuple.

        Parameters
        ----------
        idx : int
            the index of the image to read

        Returns
        -------
        TrainingPair
            the transformed (sample, target) tuple. The actual types and values of
            the components depend on the applied transforms. Usually two tensors
            should be returned as sample and target

        """
        path = os.path.join(self.root_dir, self.image_paths[idx])
        sample = self.loader.load(path)
        target = sample.copy()

        if self.transforms:
            sample = self.transforms(sample)
        if self.target_transforms:
            target = self.target_transforms(target)

        return TrainingPair(sample=sample, target=target)

    def __len__(self):
        """Return the number of images in this dataset."""
        return len(self.image_paths)

    def _construct_image_paths(self):
        """Scan root_dir and store filenames in image_paths attribute."""
        self.image_paths = [
            path if is_image_file(path) else None for path in os.listdir(self.root_dir)
        ]
        if None in self.image_paths:
            warnings.warn("At least one non-image file was found. Ignoring it")
        self.image_paths = list(filter(None, self.image_paths))
        if not self.image_paths:
            raise ValueError(
                "No valid image file found in directory {}. Aborting".format(
                    self.root_dir
                )
            )
        self.image_paths = sorted(self.image_paths)

    def get_subset(self, start=0, end=None):
        """Create a copy of this dataset object with a subset of images."""
        subset_image_paths = self.image_paths[start:end]
        return ImageNet(
            self.root_dir,
            image_paths=subset_image_paths,
            transforms=self.transforms,
            target_transforms=self.target_transforms,
            loader=self.loader,
        )

    def split_train_validation(self, train_length=None, train_percentage=None):
        """Create a train/validation split from this dataset object.

        Parameters
        ----------
        train_length : int
            length of the training split to return

        train_percentage : float or int
            percentage of examples to include in train split

        Returns
        -------
        tuple of `torch.utils.data.Dataset` objects
            the train and validation splits

        """
        if None not in [train_length, train_percentage]:
            raise ValueError("Use either lenght or percentage, not both")

        if train_length is not None:
            validation_length = len(self) - train_length
        elif train_percentage is not None:
            if isinstance(train_percentage, int):
                train_percentage = train_percentage / 100.0
            if not 0.0 < train_percentage <= 1.0:
                raise ValueError(
                    "The traininig percentage must be in (0.0, 1.0] or (0, 100]"
                )
            train_length = int(len(self) * train_percentage)
            validation_length = len(self) - train_length

        return random_split(self, (train_length, validation_length))
