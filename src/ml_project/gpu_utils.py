"""Utilities for GPU statistics logging."""
# pylint: disable=no-member
import os
import warnings
import weakref

import torch
from py3nvml.py3nvml import (
    NVML_TEMPERATURE_GPU,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetPciInfo,
    nvmlDeviceGetTemperature,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown,
)

env_cuda_dev_order = "CUDA_DEVICE_ORDER"
env_cuda_visible_devs = "CUDA_VISIBLE_DEVICES"
expected_dev_order = "PCI_BUS_ID"

# pci_info: bus, busId, device, domain, pciDeviceId, pciSubSystemId


def _torch_gpu_index_to_nvml_handle(index=None):
    """Convert the GPU index from torch to an NVML handle.

    With this function, we are sure to obtain the correct handle for the GPU
    used by pytorch.
    """
    if index is None:
        index = torch.cuda.current_device()

    device_count = nvmlDeviceGetCount()

    device_orders = os.environ.get(env_cuda_dev_order)
    if device_count > 1 and (
        device_orders is None or device_orders != expected_dev_order
    ):
        warnings.warn(
            "The environment variable {} should be set with value {}".format(
                env_cuda_dev_order, expected_dev_order
            )
        )
        warnings.warn("GPU statistics can be wrong")

    devices_by_bus_id = []
    for nvml_device_index in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(nvml_device_index)
        pci_info = nvmlDeviceGetPciInfo(handle)
        devices_by_bus_id.append((pci_info.bus, handle))

    # sort by bus id and keep only the handles
    devices_by_bus_id = [dev[1] for dev in sorted(devices_by_bus_id)]

    visible_devices = os.environ.get(env_cuda_visible_devs)
    if visible_devices is None:
        available_device_handles = devices_by_bus_id
    else:
        available_device_handles = [
            devices_by_bus_id[int(d)] for d in visible_devices.split(",")
        ]

    return available_device_handles[index]


class GpuStats:
    """Keep track of GPU utilization, when available."""

    def __init__(self, index=None):
        self.no_gpu = False

        if not torch.cuda.is_available():
            print("GPU not found. GPU stats not available")
            self.no_gpu = True
            return

        nvmlInit()
        self._finalizer = weakref.finalize(self, nvmlShutdown)

        self.handle = _torch_gpu_index_to_nvml_handle(index)

    def get_gpu_stats(self):
        """
        Return some statistics for the gpu associated with handle.

        The statistics returned are:
        - used memory in MB
        - gpu utilization percentage
        - temperature in Celsius degrees
        """
        mem = nvmlDeviceGetMemoryInfo(self.handle)
        rates = nvmlDeviceGetUtilizationRates(self.handle)
        temp = nvmlDeviceGetTemperature(self.handle, NVML_TEMPERATURE_GPU)

        return (mem.used / 1024 / 1024, rates.gpu, temp)

    def close(self):
        """Free resources occupied by nvml."""
        if not self.no_gpu:
            self._finalizer()
