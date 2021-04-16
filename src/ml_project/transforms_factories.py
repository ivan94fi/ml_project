"""This module collects the factories to create transforms in main module."""

from torchvision.transforms import Lambda, RandomCrop, ToTensor

from ml_project.transforms import (
    BrownGaussianNoise,
    ComposeCopies,
    PoissonNoise,
    RandomImpulseNoise,
    ResizeIfTooSmall,
    TextualNoise,
    WhiteGaussianNoise,
    no_op_transform,
)


class NoiseTransformCreator:
    """Create the appropriate noise transforms for the given configs."""

    def __init__(self, config):
        self.config = config

    def noise_factory_method(self):
        """Return the correct function to constuct the noise transform."""
        noise_type = self.config.noise_type
        if noise_type == "gaussian":
            return _construct_gaussian_noise
        if noise_type == "poisson":
            return _construct_poisson_noise
        if noise_type == "textual":
            return _construct_textual_noise
        if noise_type == "random_impulse":
            return _construct_random_impulse_noise

        raise ValueError("Noise type unknown")

    def create(self):
        """Get the noise transform chosen by looking at configs."""
        constructor = self.noise_factory_method()

        return constructor(self.config)


def _construct_gaussian_noise(config):
    noise_transform = {}

    if config.command == "train":
        train_params = tuple(val / 255.0 for val in config.train_params)
        val_param = config.val_param / 255.0
        if config.brown_gaussian_std is None:
            noise_transform["train"] = lambda: WhiteGaussianNoise(std=train_params)
            noise_transform["val"] = lambda: WhiteGaussianNoise(std=val_param)
        else:
            noise_transform["train"] = lambda: BrownGaussianNoise(
                kernel_std=config.brown_gaussian_std, std=train_params
            )
            noise_transform["val"] = lambda: BrownGaussianNoise(
                kernel_std=config.brown_gaussian_std, std=val_param
            )
    else:
        test_param = config.test_param / 255.0
        if config.brown_gaussian_std is None:
            noise_transform["test"] = lambda: WhiteGaussianNoise(std=test_param)
        else:
            noise_transform["test"] = lambda: BrownGaussianNoise(
                kernel_std=config.brown_gaussian_std, std=test_param
            )

    return noise_transform


def _construct_poisson_noise(config):
    noise_transform = {}
    if config.command == "train":
        noise_transform["train"] = lambda: PoissonNoise(lmbda=config.train_params)
        noise_transform["val"] = lambda: PoissonNoise(lmbda=config.val_param)
    else:
        noise_transform["test"] = lambda: PoissonNoise(lmbda=config.test_param)

    return noise_transform


def _construct_textual_noise(config):
    noise_transform = {}
    if config.command == "train":
        noise_transform["train"] = lambda: TextualNoise(coverage=config.train_params)
        noise_transform["val"] = lambda: TextualNoise(coverage=config.val_param)
    else:
        noise_transform["test"] = lambda: TextualNoise(coverage=config.test_param)

    return noise_transform


def _construct_random_impulse_noise(config):
    noise_transform = {}
    if config.command == "train":
        noise_transform["train"] = lambda: RandomImpulseNoise(p=config.train_params)
        noise_transform["val"] = lambda: RandomImpulseNoise(p=config.val_param)
    else:
        noise_transform["test"] = lambda: RandomImpulseNoise(p=config.test_param)

    return noise_transform


class TransformCreator:
    """Create the appropriate transforms for the given configs."""

    def __init__(self, config):
        self.config = config

    def create_train_transforms(self, noise_transform):
        """Train transforms factory method."""
        if self.config.train_mode == "n2n":
            target_noise = noise_transform["train"]()
        else:
            target_noise = no_op_transform

        # This noise type is applied on PIL images. Delay ToTensor after the transform
        if self.config.noise_type == "textual":
            transforms = {
                "common": ComposeCopies(
                    ResizeIfTooSmall(
                        size=self.config.input_size, stretch=self.config.stretch
                    ),
                    RandomCrop(size=self.config.input_size),
                ),
                "sample": ComposeCopies(
                    noise_transform["train"](),
                    ToTensor(),
                    Lambda(lambda sample: sample - 0.5),
                ),
                "target": ComposeCopies(
                    target_noise, ToTensor(), Lambda(lambda sample: sample - 0.5),
                ),
            }
        else:
            transforms = {
                "common": ComposeCopies(
                    ResizeIfTooSmall(
                        size=self.config.input_size, stretch=self.config.stretch
                    ),
                    RandomCrop(size=self.config.input_size),
                    ToTensor(),
                    Lambda(
                        lambda sample: sample - 0.5
                    ),  # move the tensors in [-0.5, 0.5]
                ),
                "sample": ComposeCopies(noise_transform["train"]()),
                "target": ComposeCopies(target_noise),
            }

        return transforms

    def create_val_transforms(self, transforms, noise_transform):
        """Val transforms factory method."""
        if self.config.noise_type == "textual":
            val_transforms = {
                "common": ComposeCopies(transforms["common"]),
                "sample": ComposeCopies(
                    noise_transform["val"](),
                    ToTensor(),
                    Lambda(lambda sample: sample - 0.5),
                ),
                "target": ComposeCopies(
                    ToTensor(), Lambda(lambda sample: sample - 0.5)
                ),
            }
        else:
            val_transforms = {
                "common": ComposeCopies(transforms["common"]),
                "sample": ComposeCopies(noise_transform["val"]()),
                "target": None,
            }

        return val_transforms

    def create_test_transforms(self, noise_transform):
        """Test transforms factory method."""
        if self.config.noise_type == "textual":
            test_transforms = {
                "common": None,
                "sample": ComposeCopies(
                    noise_transform["test"](),
                    ToTensor(),
                    Lambda(lambda sample: sample - 0.5),
                ),
                "target": ComposeCopies(
                    ToTensor(), Lambda(lambda sample: sample - 0.5),
                ),
            }
        else:
            test_transforms = {
                "common": ComposeCopies(
                    ToTensor(), Lambda(lambda sample: sample - 0.5)
                ),
                "sample": ComposeCopies(noise_transform["test"]()),
                "target": None,
            }

        return test_transforms

    def create(self, noise_transform):
        """Get the transforms chosen by looking at configs and noise."""
        if self.config.command == "test":
            return self.create_test_transforms(noise_transform)

        train_transforms = self.create_train_transforms(noise_transform)
        val_transforms = self.create_val_transforms(train_transforms, noise_transform)

        return train_transforms, val_transforms


class ExternalValidationTransformCreator(TransformCreator):
    """External validation transform creator."""

    def create_val_transforms(self, transforms, noise_transform):
        """Override default validation transforms with ones for external val."""
        if self.config.noise_type == "textual":
            val_transforms = {
                "common": None,
                "sample": ComposeCopies(
                    noise_transform["val"](),
                    ToTensor(),
                    Lambda(lambda sample: sample - 0.5),
                ),
                "target": ComposeCopies(
                    ToTensor(), Lambda(lambda sample: sample - 0.5)
                ),
            }
        else:
            val_transforms = {
                "common": ComposeCopies(
                    ToTensor(), Lambda(lambda sample: sample - 0.5)
                ),
                "sample": noise_transform["val"](),
                "target": None,
            }

        return val_transforms
