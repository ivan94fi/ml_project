"""Main script of the project; all computations start from here."""
import math
import time
from functools import partial

import torch
from torch.utils.data import DataLoader

from ml_project.config_parser import (
    check_directory_structure,
    define_parser,
    directory_structure,
    get_config,
    save_dict,
    tabulate_config,
)
from ml_project.datasets import ImageFolderDataset
from ml_project.losses import AnnealedL0Loss
from ml_project.models import UNet
from ml_project.procedures import test, train
from ml_project.transforms_factories import (
    ExternalValidationTransformCreator,
    NoiseTransformCreator,
    TransformCreator,
)
from ml_project.utils import get_lr_dampening_factor, set_external_seeds

complete_start = time.time()

parser = define_parser()
config = get_config(parser)

check_directory_structure(config)

print("Selected configuration")
print("=" * 60)
print(tabulate_config(config))
print("=" * 60)

noise_transform_creator = NoiseTransformCreator(config)

if config.command == "train":  # noqa: C901
    if config.fixed_seeds:
        print("Using fixed seeds for RNGs")
        torch.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        set_external_seeds()

    save_dict(config, directory_structure.RUN_CONFIG_PATH)

    if config.use_external_validation:
        transforms_creator = ExternalValidationTransformCreator(config)
    else:
        transforms_creator = TransformCreator(config)

    noise_transform = noise_transform_creator.create()
    transforms, val_transforms = transforms_creator.create(noise_transform)

    print("Dataset root:", config.dataset_root)
    full_dataset = ImageFolderDataset(config.dataset_root, transforms=transforms)
    dataset = full_dataset.get_subset(end=config.num_examples)

    if config.use_external_validation:
        print("Using external validation dataset:", config.val_dataset_root)

        train_dataset = dataset
        validation_dataset = ImageFolderDataset(
            config.val_dataset_root, transforms=val_transforms
        )
    else:
        train_dataset, validation_dataset = dataset.split_train_validation(
            train_percentage=config.train_percentage, val_transforms=val_transforms
        )

    batch_sizes = {"train": config.batch_size, "val": 1}
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
        worker_init_fn=set_external_seeds,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_sizes["val"],
        shuffle=False,
        num_workers=config.workers,
        pin_memory=config.pin_memory,
        worker_init_fn=set_external_seeds,
    )

    net = UNet().to(config.device)

    if config.noise_type == "textual":
        criterion = torch.nn.L1Loss()
    elif config.noise_type == "random_inpulse":
        criterion = AnnealedL0Loss(config.epochs)
    else:
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
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_scheduling_function
        )
    elif config.lr_scheduling_method == "reduce_on_plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )
    else:
        raise ValueError("Scheduling method not supported")

    config.starting_epoch = 1
    if config.start_from_checkpoint is not None:
        print("Restore checkpoint " + config.start_from_checkpoint)
        checkpoint = torch.load(config.start_from_checkpoint)
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["opt"])
        lr_scheduler.load_state_dict(checkpoint["sched"])
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
    print("Using test dataset:", config.test_dataset_root)

    transforms_creator = TransformCreator(config)
    noise_transform = noise_transform_creator.create()
    test_transforms = transforms_creator.create(noise_transform)

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
        worker_init_fn=set_external_seeds,
    )

    net = UNet().to(config.device)
    net.load_state_dict(checkpoint["net"])

    # Just to print the epoch used in logging
    config.starting_epoch = checkpoint["epoch"]

    if config.noise_type == "textual":
        criterion = torch.nn.L1Loss()
    if config.noise_type == "random_inpulse":
        criterion = AnnealedL0Loss(config.epochs)
    else:
        criterion = torch.nn.MSELoss()

    test_loop_start = time.time()
    test(test_dataloader, net, criterion, config)
    test_loop_end = time.time()
    print()
    print("Test time: {:.5f}".format(test_loop_end - test_loop_start))
    print("Complete script time: {:.5f}".format(test_loop_end - complete_start))
