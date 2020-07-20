"""Main script of the project; all computations start from here."""

import math
import time
from pprint import pprint

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, RandomCrop, ToTensor

from ml_project.config_parser import parse_config
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
"""
# =========================================

config = parse_config()

print("Selected configuration")
print("=" * 60)
pprint(vars(config))
print("torchvision image backend: {}".format(torchvision.get_image_backend()))
print("=" * 60)

num_examples = 20
epochs = 2
input_size = 255
stretch = False
batch_size = 4
shuffle = True
num_workers = 0
pin_memory = True

transforms = Compose(
    [
        ResizeIfTooSmall(size=input_size, stretch=stretch),
        RandomCrop(size=input_size),
        ToTensor(),
        Lambda(lambda sample: sample - 0.5),
    ]
)

full_dataset = ImageNet(
    "/home/ivan94fi/Downloads/TIXATI/ILSVRC2012_img_val", transforms=transforms
)
dataset = full_dataset.get_subset(end=num_examples)

train_dataset, validation_dataset = dataset.split_train_validation(train_percentage=0.8)
print("train dataset size:", len(train_dataset))
print("validation dataset size:", len(validation_dataset))

dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

train_loop_start = time.time()
last_batch_index = math.floor(len(train_dataset) / batch_size) - 1
print("last_batch_index:", last_batch_index)
for epoch in range(epochs):
    print("epoch:", epoch)
    for batch_index, sample in enumerate(dataloader):
        print(batch_index, end=" " if batch_index != last_batch_index else "\n")
        assert isinstance(sample, torch.Tensor), "sample is not torch tensor"
        assert sample.dtype == torch.float32, "sample is not float"
        actual_batch_size = len(sample)
        assert sample.shape == (
            actual_batch_size,
            3,
            255,
            255,
        ), "sample has wrong shape"

train_loop_end = time.time()
print("train loop time: {:.5f}".format(train_loop_end - train_loop_start))
print("complete script time: {:.5f}".format(train_loop_end - complete_start))

if config.command == "train":
    train(config)
elif config.command == "test":
    test(config)
