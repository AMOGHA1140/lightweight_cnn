"""Top-down FPN neck for the OBB detector.

1x1 lateral conv -> nearest-upsample + add -> 3x3 smooth, over the backbone's
three feature levels (C3/C4/C5).
"""

import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
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
