# pylint: disable=too-many-arguments,arguments-differ,too-many-instance-attributes
"""Classes that define the neural network models/layers for the project."""
import math
from functools import partial

import torch
import torch.nn as nn


def init_weights(module, negative_slope=0):
    """Initialize weights using He et al. (2015)."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight.data, negative_slope)
        module.bias.data.zero_()


class ConvRelu(nn.Module):
    """A layer composed of a convolution and a [Leaky]ReLU activation.

    Parameters
    ----------
    in_channels : int
        input channels for the convolutional block
    out_channels : int
        output channels for the convolutional block
    kernel_size : int or tuple
        size of the convolution kernel
    stride : int or tuple
        stride of the convolution
    padding : int or tuple
        padding of the convolution
    negative_slope : float (optional)
        if None, a ReLU is used; else a LeakyReLU with the given slope is used
        (the default is None).

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        negative_slope=None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        if negative_slope is None:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x):
        """Apply the convolution and relu to the input."""
        return self.relu(self.conv(x))


class UNet(nn.Module):
    """U-Net model for Noise2Noise training.

    Original paper for U-net:
        Ronneberger et al., 2015 https://doi.org/10.1007/978-3-319-24574-4_28

    A description of the network architecture implemented by the authors of
    Noise2Noise approach can be found in appendix A of their paper: this differs
    with Ronneberger et al. model for size and mostly because it uses fixed,
    non-learnable upsample instead of transposed convolutions in the expanding
    path.


    Parameters
    ----------
    in_channels : int
        input channels for the model (the default is 3).
    out_channels : int
        output channels for the model (the default is 3).
    expand_strategy : one of "upsample" or "transpose_convolution"
        the layers to use for the expanding path (the default is "upsample").

    """

    expand_strategies = ["upsample", "transpose_convolution"]

    def __init__(
        self, in_channels=3, out_channels=3, expand_strategy="upsample", slope=0.1
    ):
        super().__init__()

        if expand_strategy not in self.expand_strategies:
            raise ValueError(
                "Expand strategy should be one of: {}".format(self.expand_strategies)
            )

        self.expand_strategy = expand_strategy

        # Contracting path - Encoder

        # ENC_CONV0, ENC_CONV1, POOL1
        self.enc_block1 = nn.Sequential(
            ConvRelu(in_channels, 48, 3, stride=1, padding=1, negative_slope=slope),
            ConvRelu(48, 48, 3, stride=1, padding=1, negative_slope=slope),
            nn.MaxPool2d(2),
        )

        # ENC_CONV2, POOL2
        self.enc_block2 = nn.Sequential(
            ConvRelu(48, 48, 3, stride=1, padding=1, negative_slope=slope),
            nn.MaxPool2d(2),
        )

        # ENC_CONV3, POOL3
        self.enc_block3 = nn.Sequential(
            ConvRelu(48, 48, 3, stride=1, padding=1, negative_slope=slope),
            nn.MaxPool2d(2),
        )

        # ENC_CONV4, POOL4
        self.enc_block4 = nn.Sequential(
            ConvRelu(48, 48, 3, stride=1, padding=1, negative_slope=slope),
            nn.MaxPool2d(2),
        )

        # ENC_CONV5, POOL5
        self.enc_block5 = nn.Sequential(
            ConvRelu(48, 48, 3, stride=1, padding=1, negative_slope=slope),
            nn.MaxPool2d(2),
        )

        # ENC_CONV6
        self.enc_block6 = ConvRelu(48, 48, 3, stride=1, padding=1, negative_slope=slope)

        # --------------------------------------------------- #

        # Expanding path - Decoder

        # UPSAMPLE5
        self.upsample5 = self._get_upsample_layer(48, 48)

        # CONCAT5 (with POOL4) - implemented in self.forward

        # DEC_CONV5A, DEC_CONV5B, UPSAMPLE4
        self.dec_block1 = nn.Sequential(
            ConvRelu(96, 96, 3, stride=1, padding=1, negative_slope=slope),
            ConvRelu(96, 96, 3, stride=1, padding=1, negative_slope=slope),
            self._get_upsample_layer(96, 96),
        )

        # CONCAT4 (with POOL3) - implemented in self.forward

        # DEC_CONV4A, DEC_CONV4B, UPSAMPLE3
        self.dec_block2 = nn.Sequential(
            ConvRelu(144, 96, 3, stride=1, padding=1, negative_slope=slope),
            ConvRelu(96, 96, 3, stride=1, padding=1, negative_slope=slope),
            self._get_upsample_layer(96, 96),
        )

        # CONCAT3 (with POOL2) - implemented in self.forward

        # DEC_CONV3A, DEC_CONV3B, UPSAMPLE2
        self.dec_block3 = nn.Sequential(
            ConvRelu(144, 96, 3, stride=1, padding=1, negative_slope=slope),
            ConvRelu(96, 96, 3, stride=1, padding=1, negative_slope=slope),
            self._get_upsample_layer(96, 96),
        )

        # CONCAT2 (with POOL1) - implemented in self.forward

        # DEC_CONV2A, DEC_CONV2B, UPSAMPLE1
        self.dec_block4 = nn.Sequential(
            ConvRelu(144, 96, 3, stride=1, padding=1, negative_slope=slope),
            ConvRelu(96, 96, 3, stride=1, padding=1, negative_slope=slope),
            self._get_upsample_layer(96, 96),
        )

        # CONCAT1 (with input) - implemented in self.forward

        # DEC_CONV1A, DEC_CONV1B, DEC_CONV1C
        self.dec_block5 = nn.Sequential(
            ConvRelu(
                96 + in_channels, 64, 3, stride=1, padding=1, negative_slope=slope
            ),
            ConvRelu(64, 32, 3, stride=1, padding=1, negative_slope=slope),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
        )

        # Initialize weights
        # 1) common init
        self.apply(partial(init_weights, negative_slope=slope))
        # 2) last layer init (it is still He init but with gain 1.0)
        last_layer_weights = self.dec_block5[-1].weight.data
        last_layer_fan_in = last_layer_weights.numel()
        with torch.no_grad():
            last_layer_weights.normal_(0, 1.0 / math.sqrt(last_layer_fan_in))

    def _get_upsample_layer(self, in_channels=None, out_channels=None):
        """Construct the appropriate upsampling layer."""
        if self.expand_strategy == "upsample":
            return nn.Upsample(scale_factor=2, mode="nearest")
        elif self.expand_strategy == "transpose_convolution":
            return nn.ConvTranspose2d(
                in_channels, out_channels, 3, stride=2, padding=1, output_padding=1
            )
        else:
            raise ValueError("Unkown expand strategy")

    def forward(self, x):
        """Execute forward pass through the network."""
        skips = {}
        skips["input"] = x

        # Encoder
        skips["pool1_out"] = self.enc_block1(x)
        skips["pool2_out"] = self.enc_block2(skips["pool1_out"])
        skips["pool3_out"] = self.enc_block3(skips["pool2_out"])
        skips["pool4_out"] = self.enc_block4(skips["pool3_out"])
        output = self.enc_block5(skips["pool4_out"])
        output = self.enc_block6(output)

        # Decoder
        output = self.upsample5(output)
        output = self.dec_block1(torch.cat([skips["pool4_out"], output], dim=1))
        output = self.dec_block2(torch.cat([skips["pool3_out"], output], dim=1))
        output = self.dec_block3(torch.cat([skips["pool2_out"], output], dim=1))
        output = self.dec_block4(torch.cat([skips["pool1_out"], output], dim=1))
        output = self.dec_block5(torch.cat([skips["input"], output], dim=1))

        return output
