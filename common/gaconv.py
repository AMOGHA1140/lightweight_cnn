"""Geometric Adaptive Convolution (GAConv).

A drop-in replacement for a 3x3 conv that adapts its sampling pattern to the
local geometry of features. At every spatial location it predicts three
geometric parameters -- orientation ``theta`` and two extents
``sigma_major`` / ``sigma_minor`` -- builds the affine matrix
``A = R(theta) @ diag(sigma_major, sigma_minor)`` and uses it to displace a
base 3x3 grid into geometrically-structured offsets. Those offsets drive a
depthwise deformable conv, followed by a pointwise projection for channel
mixing. This handles rotation, scale and aspect ratio in one operation.

The deformable conv is genuinely depthwise: torchvision's ``deform_conv2d``
infers ``groups`` from the weight shape, so a ``[C, 1, 3, 3]`` weight gives one
filter per channel. A single offset group (18 channels) is shared across all
channels, i.e. one geometry per location. With identity init (theta=0,
sigma=1) the offsets are zero and GAConv reduces exactly to a standard
depthwise 3x3, so it starts neutral and learns to deviate.
"""

import math

import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d


class GAConv(nn.Module):
    """Geometric Adaptive Convolution.

    Args:
        channels: number of input/output channels.
        sigma_range: ``(min, max)`` bounds for the predicted extents.
    """

    def __init__(self, channels, sigma_range=(0.5, 5.0)):
        super().__init__()
        self.channels = channels
        self.sigma_min, self.sigma_max = sigma_range

        self.geometry_pred = nn.Conv2d(channels, 3, 1, bias=True)
        self.dw_weight = nn.Parameter(torch.empty(channels, 1, 3, 3))
        self.pw_proj = nn.Conv2d(channels, channels, 1, bias=False)

        # Base 3x3 grid in (row=y, col=x) order, row-major over (kh, kw). This
        # matches torchvision's offset layout: per kernel cell, (dy, dx).
        base_grid = torch.tensor([
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1], [0, 0], [0, 1],
            [1, -1], [1, 0], [1, 1],
        ], dtype=torch.float32)
        self.register_buffer("base_grid", base_grid)

        self._init_weights()

    def _init_weights(self):
        # Geometry predictor outputs the identity transform at init: zero
        # weights so the output is the bias everywhere. theta bias 0 -> theta 0;
        # sigma bias chosen so sigmoid(b)*(max-min)+min == 1.0.
        nn.init.zeros_(self.geometry_pred.weight)
        s = (1.0 - self.sigma_min) / (self.sigma_max - self.sigma_min)
        sigma_bias = math.log(s / (1.0 - s))
        with torch.no_grad():
            self.geometry_pred.bias.zero_()
            self.geometry_pred.bias[1] = sigma_bias
            self.geometry_pred.bias[2] = sigma_bias

        nn.init.kaiming_normal_(self.dw_weight, mode="fan_out")
        nn.init.kaiming_normal_(self.pw_proj.weight, mode="fan_out")

    def _activate_geometry(self, raw):
        """Map raw predictions to bounded (theta, sigma_major, sigma_minor)."""
        theta = torch.tanh(raw[:, 0:1]) * math.pi
        span = self.sigma_max - self.sigma_min
        sigma_maj = torch.sigmoid(raw[:, 1:2]) * span + self.sigma_min
        sigma_min = torch.sigmoid(raw[:, 2:3]) * span + self.sigma_min
        sigma_min = torch.minimum(sigma_min, sigma_maj)
        return theta, sigma_maj, sigma_min

    def _compute_offsets(self, theta, sigma_maj, sigma_min):
        """Build [B, 18, H, W] deformable offsets from the geometry maps."""
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        # gy: vertical (row) base coord, gx: horizontal (col) base coord.
        gy = self.base_grid[:, 0].view(1, 9, 1, 1)
        gx = self.base_grid[:, 1].view(1, 9, 1, 1)

        # Apply A = R(theta) @ diag(sigma_maj, sigma_min) to (gx, gy).
        tx = cos_t * sigma_maj * gx - sin_t * sigma_min * gy
        ty = sin_t * sigma_maj * gx + cos_t * sigma_min * gy

        # Offsets are added on top of the standard grid, so pass the delta.
        ox = tx - gx
        oy = ty - gy

        # Interleave per cell as (dy, dx) -> [B, 18, H, W].
        offsets = torch.stack([oy, ox], dim=2).reshape(oy.shape[0], 18, oy.shape[2], oy.shape[3])
        return offsets

    def geometry(self, x):
        """Activated (theta, sigma_major, sigma_minor) maps, for visualization."""
        return self._activate_geometry(self.geometry_pred(x))

    def forward(self, x):
        theta, sigma_maj, sigma_min = self._activate_geometry(self.geometry_pred(x))
        offsets = self._compute_offsets(theta, sigma_maj, sigma_min)
        x = deform_conv2d(x, offsets, self.dw_weight, padding=1)
        return self.pw_proj(x)
