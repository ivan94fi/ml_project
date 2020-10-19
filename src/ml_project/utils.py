# flake8: noqa
# pylint: skip-file
import math

import torch
from tqdm import tqdm


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


class ProgressPrinter:
    """Utility class to handle a tqdm progress bar."""

    def __init__(self, config, progress_template):
        super().__init__()
        self.config = config
        self.progress_template = progress_template

    def reset(self, phase):
        self.phase = phase
        self.batch_index = None
        self.batch_size = None

        self.progress_bar = tqdm(
            total=self.config.dataset_sizes[phase],
            dynamic_ncols=True,
            disable=not self.config.progress_bar or phase == "val",
        )

    def _should_print(self):
        """
        Check if the progress should be printed.

        Return True if the following conditions are met:
        - the phase is train
        - batch index is a multiple of print interval or we are on the last batch
        """
        if self.phase != "train":
            return False
        return (
            self.batch_index % self.config.print_interval
            == self.config.print_interval - 1
            or self.batch_index == self.config.batch_numbers["train"] - 1
        )

    def show_epoch_progress(self, *metrics):
        """Print info on metrics. If self.progress_bar is disabled, use a simple print to stdout."""
        if self._should_print():
            metrics_str = self.progress_template.format(*metrics)
            if self.progress_bar.disable:
                print(
                    "[{}/{}] ".format(
                        self.batch_index * self.config.batch_size + self.batch_size,
                        self.config.dataset_sizes[self.phase],
                    )
                    + metrics_str
                )
            else:
                self.progress_bar.set_postfix_str(metrics_str)

    def update_bar(self, n):
        self.progress_bar.update(n)

    def close_bar(self):
        self.progress_bar.close()

    def update_batch_info(self, batch_size, batch_index):
        self.batch_size = batch_size
        self.batch_index = batch_index


def calculate_psnr(image, target, data_range=1.0, eps=1e-8):
    """Compute PSNR between two minibatches of images (image and target).

    The minibatches are expected to have shape (batch_size, channels, width, height).

    Parameters
    ----------
    image : torch.Tensor
        Noisy images batch
    target : torch.Tensor
        Clean images batch
    data_range : int or float
        The value range of input images (the default is 1.0)
    eps : float
        Small constant added to log argument for numerical stability (the
        default is 1e-8)

    Returns
    -------
    float
        PSNR measure between the input images

    """
    mse = ((image - target) ** 2).mean().item()
    return psnr_from_mse(mse, data_range, eps)


def psnr_from_mse(mse, data_range=1.0, eps=1e-8):
    """Compute the PSNR from the given mean square error value."""
    return 10.0 * math.log10(eps + (data_range ** 2) / mse)


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
