"""Main script of the project; all computations start from here."""

import math
import time

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, RandomCrop, ToTensor

from ml_project.config_parser import parse_config, tabulate_config
from ml_project.datasets import ImageNet
from ml_project.procedures import test, train
from ml_project.transforms import ResizeIfTooSmall

complete_start = time.time()

# =========================================
"""
TODO:
* load targets
* add noise to samples and targets
* train and test functions (loops inside)
* noise in cli arguments
* choose dataset
"""
# =========================================

config = parse_config()
config.torchvision_image_backend = torchvision.get_image_backend()

print("Selected configuration")
print("=" * 60)
print(tabulate_config(config))
print("=" * 60)

transforms = Compose(
    [
        ResizeIfTooSmall(size=config.input_size, stretch=config.stretch),
        RandomCrop(size=config.input_size),
        ToTensor(),
        Lambda(lambda sample: sample - 0.5),  # move the tensors in [-0.5, 0.5]
    ]
)

full_dataset = ImageNet(
    "/home/ivan94fi/Downloads/TIXATI/ILSVRC2012_img_val", transforms=transforms
)
dataset = full_dataset.get_subset(end=config.num_examples)

train_dataset, validation_dataset = dataset.split_train_validation(
    train_percentage=config.train_percentage
)
print("train dataset size:", len(train_dataset))
print("validation dataset size:", len(validation_dataset))

dataloader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=config.shuffle,
    num_workers=config.workers,
    pin_memory=config.pin_memory,
)

if config.dry_run:
    import sys

    print("Dry run. Not executing training")
    sys.exit()

train_loop_start = time.time()
last_batch_index = math.floor(len(train_dataset) / config.batch_size) - 1
for epoch in range(config.epochs):
    print("epoch:", epoch)
    for batch_index, sample in enumerate(dataloader):
        print(batch_index, end=" " if batch_index != last_batch_index else "\n")
        assert isinstance(sample, torch.Tensor), "sample is not torch tensor"
        assert sample.dtype == torch.float32, "sample is not float"
        actual_batch_size = len(sample)
        assert sample.shape == (
            actual_batch_size,
            3,
            config.input_size,
            config.input_size,
        ), "sample has wrong shape"

train_loop_end = time.time()
print("train loop time: {:.5f}".format(train_loop_end - train_loop_start))
print("complete script time: {:.5f}".format(train_loop_end - complete_start))

if config.command == "train":
    train(config)
elif config.command == "test":
    test(config)
