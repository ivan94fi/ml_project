"""Classes to load images from the used datasets."""

import os
import warnings
from collections import namedtuple

import torch
from torch.utils.data import Dataset

from ml_project.image_loaders import PillowLoader


def is_image_file(filename):
    """Check whether the given path refers to an image file using its exstension."""
    return os.path.splitext(filename)[1].lower() in [".jpg", ".jpeg", ".png"]


def read_directory_from_env(env_var_name):
    """Try to read a directory name from an environment variable."""
    try:
        directory = os.environ[env_var_name]
        return directory
    except KeyError as key_error:
        raise ValueError(
            "Environment variable {} not set. Set it with before running the script "
            "or use the appropriate cli option".format(env_var_name)
        ) from key_error


TrainingPair = namedtuple("TrainingPair", ["sample", "target"])

ENV_VAR_DATASET_ROOT = "NOISE2NOISE_DATASET_ROOT"
ENV_VAR_VALIDATION_DATASET_ROOT = "NOISE2NOISE_VALIDATION_DATASET_ROOT"
ENV_VAR_TEST_DATASET_ROOT = "NOISE2NOISE_TEST_DATASET_ROOT"


class ImageFolderDataset(Dataset):
    """Custom torch.utils.data.Dataset subclass for loading images.

    Parameters
    ----------
    root_dir : str
        Path to dataset directory (where extracted jpg images reside). If it is
        None, attempt to read the environment variable NOISE2NOISE_DATASET_ROOT
    image_paths : list of str, optional
        If passed, use these filenames to contruct the dataset,
        do not parse the root directory (the default is None).
    transforms : dict of callables, optional
        Functions or transforms to apply to data; three keys are expected:
        "common", "sample", "target". The transforms are applied only to the
        element specified by their key and common transforms are applied first
        (the default is None).
    loader : AbstractImageLoader implementation, optional
        The image loader to use: defaults to PillowLoader

    """

    # pylint: disable=too-many-arguments
    def __init__(self, root_dir, image_paths=None, transforms=None, loader=None):
        self.root_dir = root_dir
        if not os.path.isdir(self.root_dir):
            raise ValueError("dataset directory does not point to a valid directory")

        if image_paths is None:
            self._construct_image_paths()
        else:
            self.image_paths = image_paths

        if transforms is None:
            self.transforms = {}
        else:
            self.transforms = dict(transforms)

        if loader is None:
            self.loader = PillowLoader()
        else:
            self.loader = loader

    @property
    def common_transforms(self):
        """Get common transforms from self.transforms."""
        return self.transforms.get("common")

    @property
    def sample_transforms(self):
        """Get sample transforms from self.transforms."""
        return self.transforms.get("sample")

    @property
    def target_transforms(self):
        """Get target transforms from self.transforms."""
        return self.transforms.get("target")

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

        if self.common_transforms is not None:
            sample = self.common_transforms(sample)

        if isinstance(sample, torch.Tensor):
            target = sample.detach().clone()
        else:
            target = sample.copy()

        if self.sample_transforms is not None:
            sample = self.sample_transforms(sample)
        if self.target_transforms is not None:
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

    def get_subset(self, start=0, end=None, indices=None):
        """Create a copy of this dataset object with a subset of images."""
        if indices is None:
            subset_image_paths = self.image_paths[start:end]
        else:
            subset_image_paths = [self.image_paths[i] for i in indices]
        return ImageFolderDataset(
            self.root_dir,
            image_paths=subset_image_paths,
            transforms=self.transforms,
            loader=self.loader,
        )

    def split_train_validation(
        self, train_length=None, train_percentage=None, val_transforms=None
    ):
        """Create a train/validation split from this dataset object.

        Parameters
        ----------
        train_length : int
            length of the training split to return

        train_percentage : float or int
            percentage of examples to include in train split

        val_transforms : dict of callables
            transforms for the validation split. If not given, transforms from
            the parent dataset will be applied to both splits (the default is None)

        Returns
        -------
        tuple of `ImageFolderDataset` objects
            the train and validation splits

        """
        if None not in [train_length, train_percentage]:
            raise ValueError("Use either lenght or percentage, not both")

        if train_length is None:
            if isinstance(train_percentage, int):
                train_percentage = train_percentage / 100.0
            if not 0.0 < train_percentage <= 1.0:
                raise ValueError(
                    "The traininig percentage must be in (0.0, 1.0] or (0, 100]"
                )
            train_length = int(len(self) * train_percentage)

        indices = torch.randperm(len(self)).tolist()

        train_split, validation_split = (
            self.get_subset(indices=indices[:train_length]),
            self.get_subset(indices=indices[train_length:]),
        )

        if val_transforms is not None:
            validation_split.transforms = val_transforms

        return train_split, validation_split
