"""Top-down FPN neck for the OBB detector.

1x1 lateral conv -> nearest-upsample + add -> smooth, over the backbone's three
feature levels (C3/C4/C5). The smooth stage is a standard 3x3 conv by default,
or GAConv when ``smooth_conv='gaconv'``.
"""

import torch.nn as nn
import torch.nn.functional as F

from common.gaconv import GAConv


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels=128, smooth_conv="standard"):
        super().__init__()
        if smooth_conv not in ("standard", "gaconv"):
            raise ValueError(f"smooth_conv must be 'standard' or 'gaconv', got {smooth_conv!r}")
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        if smooth_conv == "gaconv":
            self.smooth = nn.ModuleList([GAConv(out_channels) for _ in in_channels])
        else:
            self.smooth = nn.ModuleList(
                [nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels]
            )

    def forward(self, feats):
        # feats: list of feature maps, lowest-stride first.
        laterals = [lat(f) for lat, f in zip(self.lateral, feats)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], scale_factor=2, mode="nearest"
            )
        return [s(lvl) for s, lvl in zip(self.smooth, laterals)]
