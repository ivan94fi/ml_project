"""Utility functions and classes."""
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
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

from ml_project.datasets import TrainingPair


def to_channel_last(tensor):
    """Move channel axis in last position."""
    return tensor.permute(1, 2, 0)


def to_channel_first(tensor):
    """Move channel axis in first position."""
    return tensor.permute(2, 0, 1)


def prepare_for_imshow(tensor, bias=None):
    """Convert to channel last, add bias, clamp to [0, 1]."""
    result = to_channel_last(tensor)
    if bias:
        result += bias
    return result.clamp(0.0, 1.0)


def transpose(tensor):
    """Swap the last dimension with the second last."""
    if tensor.dim() < 2:
        raise ValueError("The input tensor should have at least two dimensions")
    dims = list(range(tensor.dim()))
    dims[-2:] = dims[-2:][::-1]
    return tensor.permute(dims)


def get_lr_dampening_factor(epoch, total_epochs, percentage_to_dampen):
    """
    Return the multiplicative factor used to dampen the learning rate.

    The learning rate is left unchanged until the last `percentage_to_dampen`
    percentage of training epochs. After, it is gradually decreased until it
    reaches zero at the last epoch.
    """
    percentage_to_dampen /= 100
    dampen_start_epoch = total_epochs * (1 - percentage_to_dampen)
    dampen_factor = 1.0
    if epoch >= dampen_start_epoch:
        cosine_arg = (
            (epoch - dampen_start_epoch) / percentage_to_dampen
        ) / total_epochs
        dampen_factor = (0.5 + math.cos(cosine_arg * math.pi) / 2) ** 2
    return dampen_factor


def pad(data, divisor=32, mode="reflect"):
    """
    Pad the training pair.

    The padding applied is enough to make the image width and height divisible
    for the specified divisor.

    The input data is expected to be 4-dimensional, with the first dimension of
    size 1. The first and second dimension are left untouched, pad is applied to
    the last two dimension if necessary.

    """
    if data.sample.shape[0] != 1:
        raise ValueError("The first dimension must be of size 1")
    width = data.sample.shape[2]
    height = data.sample.shape[3]

    if width % divisor != 0 or height % divisor != 0:
        padded_width = math.ceil(width / divisor) * divisor
        padded_height = math.ceil(height / divisor) * divisor
        pad_amount = (0, (padded_height - height), 0, (padded_width - width))
        sample = F.pad(data.sample, pad_amount, mode=mode)
        target = F.pad(data.target, pad_amount, mode=mode)
        data = TrainingPair(sample=sample, target=target)
    return data


def get_gaussian_kernel(stddev, dimensions=1, size=None, limit=4):
    """
    Return a gaussian kernel with the given standard deviation.

    Parameters
    ----------
    stddev : float
        Standard deviation of the gaussian kernel
    dimensions: int
        Control the dimensions of the kernel (mono or bi-dimensional)
    size : int
        Use a predefined size for the kernel. If None, compute the size as
        round(limit * stddev)
    limit : int
        Limit for kernel size

    Returns
    -------
    Torch.tensor
        The constructed (1D or 2D) gaussian kernel (normalized)

    """
    if size is not None:
        radius = size // 2
        ksize = size
    else:
        radius = round(limit * stddev)
        ksize = (radius * 2) + 1
    kernel_1d = torch.exp(
        -0.5 * ((torch.arange(ksize, dtype=torch.float) - radius) / stddev) ** 2
    )
    if dimensions == 1:
        kernel = kernel_1d
    else:
        kernel = torch.ger(kernel_1d, kernel_1d)
    return kernel / kernel.sum()


def checkpoint_fname_template():
    """Return a filename template: 'n2n_<current_timestamp>_e{}.pt'."""
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
    # pylint: disable=no-member
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
    for axis, image in zip(axes, images):
        image = prepare_for_imshow(make_grid(image.detach().clone().cpu()), 0.5)
        axis.imshow(image)
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
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

        self.phase = None
        self.batch_index = None
        self.batch_size = None
        self.progress_bar = None

        self.config = config
        self.progress_template = progress_template

    def reset(self, phase):
        """Configure for the passed phase."""
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
        """
        Print info on metrics.

        If self.progress_bar is disabled, use a simple print to stdout.

        """
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
        """Set the iteration index on the bar."""
        self.progress_bar.update(n)

    def close_bar(self):
        """Close the bar and remove the postfix string."""
        self.progress_bar.set_postfix_str("")
        self.progress_bar.close()

    def update_batch_info(self, batch_size, batch_index):
        """Update batch size as it can vary throughout iterations."""
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
