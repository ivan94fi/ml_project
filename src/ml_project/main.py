"""Main script of the project; all computations start from here."""
import math
import os
import random
import time
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda, RandomCrop, ToTensor

from ml_project.config_parser import (
    define_parser,
    parse_config,
    save_config_file,
    tabulate_config,
)
from ml_project.datasets import ImageFolderDataset
from ml_project.models import UNet
from ml_project.procedures import test, train
from ml_project.transforms import (
    BrownGaussianNoise,
    ComposeCopies,
    PoissonNoise,
    ResizeIfTooSmall,
    WhiteGaussianNoise,
)
from ml_project.utils import get_lr_dampening_factor

complete_start = time.time()

parser = define_parser()
config = parse_config(parser)

print("Selected configuration")
print("=" * 60)
print(tabulate_config(config))
print("=" * 60)


# pylint: disable=unused-argument
def _set_external_seeds(worker_id=None):
    random.seed(config.seed)
    np.random.seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)


if config.fixed_seeds:
    print("Using fixed seeds for RNGs")
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _set_external_seeds()


if config.command == "train":  # noqa: C901
    save_config_file(config)

    noise_transform = {}
    if config.noise_type == "gaussian":
        train_params = tuple(val / 255.0 for val in config.train_params)
        val_param = config.val_param / 255.0
        if config.brown_gaussian_std is not None:
            noise_transform["train"] = lambda: BrownGaussianNoise(
                kernel_std=config.brown_gaussian_std, std=train_params
            )
            noise_transform["val"] = lambda: BrownGaussianNoise(
                kernel_std=config.brown_gaussian_std, std=val_param
            )
        else:
            noise_transform["train"] = lambda: WhiteGaussianNoise(std=train_params)
            noise_transform["val"] = lambda: WhiteGaussianNoise(std=val_param)
    elif config.noise_type == "poisson":
        noise_transform["train"] = lambda: PoissonNoise(lmbda=train_params)
        noise_transform["val"] = lambda: PoissonNoise(lmbda=val_param)
    else:
        raise NotImplementedError

    batch_sizes = {"train": config.batch_size, "val": 1}

    sample_transforms = ComposeCopies([noise_transform["train"]()])
    target_transforms = ComposeCopies(
        sample_transforms if config.train_mode == "n2n" else []
    )
    common_transforms = ComposeCopies(
        [
            ResizeIfTooSmall(size=config.input_size, stretch=config.stretch),
            RandomCrop(size=config.input_size),
            ToTensor(),
            Lambda(lambda sample: sample - 0.5),  # move the tensors in [-0.5, 0.5]
        ]
    )

    transforms = {
        "common": common_transforms,
        "sample": sample_transforms,
        "target": target_transforms,
    }

    print("Dataset root:", config.dataset_root)
    full_dataset = ImageFolderDataset(config.dataset_root, transforms=transforms)
    dataset = full_dataset.get_subset(end=config.num_examples)

    if config.use_external_validation:
        print("Using external validation dataset:", config.val_dataset_root)

        train_dataset = dataset
        val_transforms = {
            "common": ComposeCopies([ToTensor(), Lambda(lambda sample: sample - 0.5)]),
            "sample": noise_transform["val"](),
            "target": None,
        }
        validation_dataset = ImageFolderDataset(
            config.val_dataset_root, transforms=val_transforms
        )
    else:
        val_transforms = {
            "common": ComposeCopies([common_transforms]),
            "sample": noise_transform["val"](),
            "target": None,
        }

        train_dataset, validation_dataset = dataset.split_train_validation(
            train_percentage=config.train_percentage, val_transforms=val_transforms
        )

    config.dataset_sizes = {"train": len(train_dataset), "val": len(validation_dataset)}
    config.batch_numbers = {
        phase: math.ceil(size / batch_sizes[phase])
        for phase, size in config.dataset_sizes.items()
    }
    print("train dataset size:", config.dataset_sizes["train"])
    print("validation dataset size:", config.dataset_sizes["val"])

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_sizes["train"],
        shuffle=config.shuffle,
        num_workers=config.workers,
        pin_memory=config.pin_memory,
        worker_init_fn=_set_external_seeds if config.fixed_seeds else None,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_sizes["val"],
        shuffle=False,
        num_workers=config.workers,
        pin_memory=config.pin_memory,
        worker_init_fn=_set_external_seeds if config.fixed_seeds else None,
    )

    net = UNet().to(config.device)

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        net.parameters(), lr=config.learning_rate, betas=(0.9, 0.99)
    )

    # total_epochs must always be the entire number of epochs: restart handled by
    # restoring scheduler
    if config.lr_scheduling_method == "cosine":
        lr_scheduling_function = partial(
            get_lr_dampening_factor,
            total_epochs=config.epochs,
            percentage_to_dampen=config.lr_dampen_percentage,
        )
    else:
        raise ValueError("Scheduling method not supported")

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scheduling_function)

    if config.start_from_checkpoint is not None:
        print("Restore checkpoint " + config.start_from_checkpoint)
        checkpoint = torch.load(config.start_from_checkpoint)
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["opt"])
        lr_scheduler.load_state_dict(checkpoint["sched"])
        print("Overriding starting epoch value with the one found in the checkpoint")
        if config.starting_epoch != 1:
            # a starting epoch is passed from cli: check that it matches the one in
            # checkpoint
            if config.starting_epoch != checkpoint["epoch"]:
                raise ValueError(
                    "Mismatch between epochs in checkpoint "
                    "and command line argument: got {} and {}.".format(
                        checkpoint["epoch"], config.starting_epoch
                    )
                )
        config.starting_epoch = checkpoint["epoch"] + 1

    dataloaders = {"train": dataloader, "val": validation_dataloader}

    print("Epochs to run: {} to {}".format(config.starting_epoch, config.epochs))
    print()
    print("Starting train loop")
    print()
    train_loop_start = time.time()
    train(dataloaders, net, criterion, optimizer, lr_scheduler, config)
    train_loop_end = time.time()
    print()
    print("Train loop time: {:.5f}".format(train_loop_end - train_loop_start))
    print("Complete script time: {:.5f}".format(train_loop_end - complete_start))

###############################################################################

elif config.command == "test":
    print("Restore checkpoint " + config.test_checkpoint)
    checkpoint = torch.load(config.test_checkpoint)
    print("Epoch from checkpoint: {}".format(checkpoint["epoch"]))

    noise_transform = {}
    if config.noise_type == "gaussian":
        test_param = config.test_param / 255.0
        if config.brown_gaussian_std is not None:
            noise_transform["test"] = lambda: BrownGaussianNoise(
                kernel_std=config.brown_gaussian_std, std=test_param
            )
        else:
            noise_transform["test"] = lambda: WhiteGaussianNoise(std=test_param)
    elif config.noise_type == "poisson":
        noise_transform["test"] = lambda: PoissonNoise(lmbda=test_param)
    else:
        raise NotImplementedError

    print("Using test dataset:", config.test_dataset_root)

    test_transforms = {
        "common": ComposeCopies([ToTensor(), Lambda(lambda sample: sample - 0.5)]),
        "sample": noise_transform["test"](),
        "target": None,
    }
    test_dataset = ImageFolderDataset(
        config.test_dataset_root, transforms=test_transforms
    )

    config.dataset_sizes = {"test": len(test_dataset)}
    config.batch_numbers = config.dataset_sizes
    print("test dataset size:", config.dataset_sizes["test"])

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=config.pin_memory,
        worker_init_fn=_set_external_seeds if config.fixed_seeds else None,
    )

    net = UNet().to(config.device)
    net.load_state_dict(checkpoint["net"])

    criterion = torch.nn.MSELoss()

    test_loop_start = time.time()
    test(test_dataloader, net, criterion, config)
    test_loop_end = time.time()
    print()
    print("Test time: {:.5f}".format(test_loop_end - test_loop_start))
    print("Complete script time: {:.5f}".format(test_loop_end - complete_start))
