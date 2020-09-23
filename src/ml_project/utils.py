# flake8: noqa
# pylint: skip-file

import torch


def to_channel_last(tensor):
    return tensor.permute(1, 2, 0)


def to_channel_first(tensor):
    return tensor.permute(2, 0, 1)


def prepare_for_imshow(tensor, bias=None):
    result = to_channel_last(tensor)
    if bias:
        result += bias
    return result.clamp(0.0, 1.0)


def uniform(lower, upper):
    return torch.rand(1).item() * (upper - lower) + lower


def should_print(phase, batch_index, config):
    """Print every config.print_interval batches and the last batch"""
    if phase != "train":
        return False
    return (
        batch_index % config.print_interval == config.print_interval - 1
        or batch_index == config.batch_numbers["train"] - 1
    )
