# flake8: noqa
# pylint: skip-file
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from py3nvml.py3nvml import (
    NVML_TEMPERATURE_GPU,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetTemperature,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown,
)
from torchvision.utils import make_grid
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


def checkpoint_fname_template():
    timestamp = datetime.now().strftime("%b%d_%H-%M")
    return "n2n_" + timestamp + "_e{}.pt"


def get_nvml_handle(index=None):
    """Return an handle to the current gpu to query some stats."""
    if index is None:
        index = torch.cuda.current_device()
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices is not None:
            visible_devices = [int(d) for d in visible_devices.split(",")]
            # The following line assumes devices ids are specified in order and
            # without gaps in CUDA_VISIBLE_DEVICES definition
            index += max(visible_devices)

    nvmlInit()
    return nvmlDeviceGetHandleByIndex(index)


def get_gpu_stats(handle):
    """
    Return some statistics for the gpu associated with handle.

    The statistics returned are:
    - used memory in MB
    - gpu utilization percentage
    - temperature in Celsius degrees
    """
    mem = nvmlDeviceGetMemoryInfo(handle)
    rates = nvmlDeviceGetUtilizationRates(handle)
    temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
    return (mem.used / 1024 / 1024, rates.gpu, temp)


def nvml_shutdown():
    """Free resources occupied by nvml."""
    nvmlShutdown()


def create_figure(images, title=None):
    """
    Create a matplotlib figure with the passed batches of tensor images.

    The input tensors are copied, each batch is flattened, then the images
    obtained are plotted in column in a matplotlib figure.
    """
    fig, axes = plt.subplots(len(images), 1)
    for ax, image in zip(axes, images):
        image = prepare_for_imshow(make_grid(image.detach().clone().cpu()), 0.5)
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    fig.suptitle(
        title,
        fontsize=15,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.5, "pad": 5},
    )
    return fig


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
        self.progress_bar.set_postfix_str("")
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
