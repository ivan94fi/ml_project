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


class _RepeatSampler:
    """Sampler that repeats forever."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.DataLoader):
    """
    Dataloader with process reuse taken from pytorch issue 15849.

    See https://github.com/pytorch/pytorch/issues/15849 for more information.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)
