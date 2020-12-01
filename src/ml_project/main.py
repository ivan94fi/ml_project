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

from ml_project.config_parser import parse_config, tabulate_config
from ml_project.datasets import ImageFolderDataset
from ml_project.models import UNet
from ml_project.procedures import test, train
from ml_project.transforms import ComposeCopies, GaussianNoise, ResizeIfTooSmall
from ml_project.utils import get_lr_dampening_factor

complete_start = time.time()

# =========================================
"""
TODO:
* test function
* other noise
"""
# =========================================

config = parse_config()

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


common_transforms = ComposeCopies(
    [
        ResizeIfTooSmall(size=config.input_size, stretch=config.stretch),
        RandomCrop(size=config.input_size),
        ToTensor(),
        Lambda(lambda sample: sample - 0.5),  # move the tensors in [-0.5, 0.5]
    ]
)

if config.noise_type == "gaussian":
    NoiseTransform = GaussianNoise
else:
    # TODO: add other types of noise
    raise NotImplementedError

config.std_range = tuple(val / 255.0 for val in config.std_range)
config.val_std /= 255.0

sample_transforms = ComposeCopies([NoiseTransform(std=config.std_range)])
target_transforms = ComposeCopies(
    sample_transforms if config.train_mode == "n2n" else []
)

transforms = {
    "common": common_transforms,
    "sample": sample_transforms,
    "target": target_transforms,
}

val_transforms = {
    "common": ComposeCopies(common_transforms),
    "sample": ComposeCopies([NoiseTransform(std=config.val_std)]),
    "target": None,
}

full_dataset = ImageFolderDataset(config.dataset_root, transforms=transforms)
dataset = full_dataset.get_subset(end=config.num_examples)

train_dataset, validation_dataset = dataset.split_train_validation(
    train_percentage=config.train_percentage, val_transforms=val_transforms
)

config.dataset_sizes = {"train": len(train_dataset), "val": len(validation_dataset)}
config.batch_numbers = {
    phase: math.ceil(size / config.batch_size)
    for phase, size in config.dataset_sizes.items()
}
print("train dataset size:", config.dataset_sizes["train"])
print("validation dataset size:", config.dataset_sizes["val"])

dataloader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=config.shuffle,
    num_workers=config.workers,
    pin_memory=config.pin_memory,
    worker_init_fn=_set_external_seeds if config.fixed_seeds else None,
)

validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=config.batch_size,
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

if config.start_from_checkpoint is not None:
    print("Restore checkpoint " + config.start_from_checkpoint)
    checkpoint = torch.load(config.start_from_checkpoint)
    net.load_state_dict(checkpoint["net"])
    optimizer.load_state_dict(checkpoint["opt"])
    starting_epoch_from_checkpoint = checkpoint["epoch"] + 1
    if config.starting_epoch == 1:
        print("Override starting epoch value with the one found in the checkpoint")
        config.starting_epoch = starting_epoch_from_checkpoint

print("Epochs to run: {} to {}".format(config.starting_epoch, config.epochs))

lr_scheduling_function = partial(
    get_lr_dampening_factor,
    total_epochs=config.epochs - config.starting_epoch + 1,
    percentage_to_dampen=50,
)

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scheduling_function)

if config.start_from_checkpoint is not None:
    scheduler_state_dict = checkpoint["sched"]
    scheduler_state_dict.pop("lr_lambdas", None)
    lr_scheduler.load_state_dict(scheduler_state_dict)

dataloaders = {"train": dataloader, "val": validation_dataloader}

if config.command == "train":
    print()
    print("Starting train loop")
    print()
    train_loop_start = time.time()
    train(dataloaders, net, criterion, optimizer, lr_scheduler, config)
    train_loop_end = time.time()
    print()
    print("Train loop time: {:.5f}".format(train_loop_end - train_loop_start))
elif config.command == "test":
    test(config)

print("Complete script time: {:.5f}".format(train_loop_end - complete_start))
