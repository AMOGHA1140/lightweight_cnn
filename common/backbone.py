"""GhostTriRemote-X Pro++: a lightweight backbone for remote-sensing detection.

Defines ``GhostTriRemoteXProPP`` and its building blocks. The attention blocks
follow their reference papers: GhostModule (GhostNet, CVPR 2020), CoordAtt
(Coordinate Attention, CVPR 2021), CBAM (ECCV 2018).
"""

import torch
import torch.nn as nn


class GhostModule(nn.Module):
    """Ghost Module (GhostNet, CVPR 2020).

    Generates intrinsic feature maps with a primary conv, then the ghost maps
    with a strictly depthwise cheap op (one filter per intrinsic channel).
    """

    def __init__(self, in_ch, out_ch, ratio=2, kernel=1, dw_kernel=3, stride=1, relu=True):
        super().__init__()
        init = max(1, int(out_ch / ratio))
        cheap = out_ch - init
        self.primary = nn.Sequential(
            nn.Conv2d(in_ch, init, kernel, stride, kernel // 2, bias=False),
            nn.BatchNorm2d(init),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )
        if cheap > 0 and init > 0:
            self.cheap = nn.Sequential(
                nn.Conv2d(init, init, dw_kernel, 1, dw_kernel // 2, groups=init, bias=False),
                nn.BatchNorm2d(init),
                nn.ReLU(inplace=True) if relu else nn.Identity(),
            )
        else:
            self.cheap = nn.Identity()
        self.cheap_channels = cheap

    def forward(self, x):
        y = self.primary(x)
        if self.cheap_channels > 0:
            ghost = self.cheap(y)
            # If init != cheap (odd out_ch), slice to match.
            if ghost.shape[1] != self.cheap_channels:
                ghost = ghost[:, :self.cheap_channels]
            return torch.cat([y, ghost], dim=1)
        return y


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """Coordinate Attention (Hou et al., CVPR 2021).
    Official: github.com/houqb/CoordAttention
    """

    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, 1, 1, 0)
        self.conv_w = nn.Conv2d(mip, oup, 1, 1, 0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_h * a_w
        return out


class MultiStripAttn(nn.Module):
    """Strip attention for long, thin objects (inspired by Strip Pooling, CVPR-20).

    Four asymmetric depth-wise convolutions: (1x7), (7x1), (1x15), (15x1).
    Output is ``x * sigmoid(h7 + w7 + h15 + w15)``.
    """

    def __init__(self, ch):
        super().__init__()
        self.h7 = nn.Conv2d(ch, ch, (1, 7), 1, (0, 3), groups=ch)
        self.w7 = nn.Conv2d(ch, ch, (7, 1), 1, (3, 0), groups=ch)
        self.h15 = nn.Conv2d(ch, ch, (1, 15), 1, (0, 7), groups=ch)
        self.w15 = nn.Conv2d(ch, ch, (15, 1), 1, (7, 0), groups=ch)

    def forward(self, x):
        a = (self.h7(x) + self.w7(x)) + (self.h15(x) + self.w15(x))
        return x * torch.sigmoid(a)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (CVPR-18)."""

    def __init__(self, ch, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, max(1, ch // r), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, ch // r), ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        s = self.pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1, 1)
        return x * s


class ChannelGate(nn.Module):
    """Channel attention with avg-pool + max-pool through a shared MLP."""

    def __init__(self, ch, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(ch, max(1, ch // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, ch // reduction), ch, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.mlp(x.mean(dim=(2, 3)))
        max_out = self.mlp(x.amax(dim=(2, 3)))
        scale = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * scale


class SpatialGate(nn.Module):
    """Spatial attention with channel-wise avg+max pool."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        scale = torch.sigmoid(self.spatial(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module (Woo et al., ECCV 2018).
    Official: github.com/Jongchan/attention-module
    """

    def __init__(self, ch, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_gate = ChannelGate(ch, reduction)
        self.spatial_gate = SpatialGate(kernel_size)

    def forward(self, x):
        x = self.channel_gate(x)
        x = self.spatial_gate(x)
        return x


class ChannelShuffleFusion(nn.Module):
    """Parameter-free channel shuffle (ShuffleNet, CVPR-18)."""

    def __init__(self, groups=4):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()
        g = self.groups
        if c % g != 0:
            return x
        x = x.view(b, g, c // g, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return x.view(b, c, h, w)


class RotationInvariantFusion(nn.Module):
    """Rotation-Invariant Feature Fusion (RIF) -- novel contribution.

    Creates 4 rotated copies (0/90/180/270 deg) and fuses them with learnable
    per-channel weights ``alpha`` of shape ``[4, C, 1, 1]`` initialised to 1/4.
    """

    def __init__(self, ch):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(4, ch, 1, 1) / 4)

    def forward(self, x):
        r0 = x
        r90 = torch.rot90(x, 1, [2, 3])
        r180 = torch.rot90(x, 2, [2, 3])
        r270 = torch.rot90(x, 3, [2, 3])
        stack = torch.stack([r0, r90, r180, r270], 0)  # 4,B,C,H,W
        fused = (self.alpha.unsqueeze(1) * stack).sum(0)
        return fused


class GhostBottle(nn.Module):
    """Composite block: Ghost(expand) -> DW conv(stride) -> Ghost(project) ->
    CoordAtt -> ChannelShuffle -> + residual."""

    def __init__(self, in_ch, mid_ch, out_ch, stride=1, use_ca=True):
        super().__init__()
        self.g1 = GhostModule(in_ch, mid_ch, relu=True)
        self.dw = nn.Conv2d(mid_ch, mid_ch, 3, stride, 1, groups=mid_ch, bias=False) if stride > 1 else nn.Identity()
        self.bn = nn.BatchNorm2d(mid_ch) if stride > 1 else nn.Identity()
        self.g2 = GhostModule(mid_ch, out_ch, relu=False)
        self.attn = CoordAtt(out_ch, out_ch) if use_ca else nn.Identity()
        self.shuffle = ChannelShuffleFusion(4)

        if stride == 1 and in_ch == out_ch:
            self.short = nn.Identity()
        else:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        res = self.short(x)
        x = self.g1(x)
        x = self.bn(self.dw(x))
        x = self.g2(x)
        x = self.attn(x)
        x = self.shuffle(x)
        return x + res


class GhostTriRemoteXProPP(nn.Module):
    """Lightweight backbone with Rotation-Invariant Fusion.

    Channel progression (width_mult=1.0): 48 -> 64 -> 128 -> 192 -> 256.

    ``forward_features`` returns three feature maps for the FPN neck:
    C3 (stride 8, 128ch), C4 (stride 16, 192ch), C5 (stride 32, 256ch).
    """

    def __init__(self, num_classes=200, drop_rate=0.2, width_mult=1.0):
        super().__init__()

        def B(x):
            return max(8, int(x * width_mult))

        # Ensure all channels are multiples of 8 for better compatibility.
        def make_divisible(v, divisor=8):
            return max(divisor, int(v + divisor / 2) // divisor * divisor)

        # Stem
        stem_ch = make_divisible(B(48))
        self.stem = GhostModule(3, stem_ch, kernel=3, stride=2)

        ch1 = make_divisible(B(64))
        ch2 = make_divisible(B(128))
        ch3 = make_divisible(B(192))
        ch4 = make_divisible(B(256))

        self.stage1 = self._make_stage(stem_ch, ch1, ch1, stride=2, n=3)
        self.stage2 = self._make_stage(ch1, ch2, ch2, stride=2, n=4)
        self.stage3 = self._make_stage(ch2, ch3, ch3, stride=2, n=4)
        self.rif = RotationInvariantFusion(ch3)
        self.stage4 = self._make_stage(ch3, ch4, ch4, stride=2, n=2)

        ch_last = ch4
        self.strip = MultiStripAttn(ch_last)
        self.cbam = CBAM(ch_last)

        # Classification head (used for backbone pretraining only).
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.fc = nn.Linear(ch_last, num_classes)

        self._init_weights()

    def _make_stage(self, in_c, mid_c, out_c, stride, n):
        layers = [GhostBottle(in_c, mid_c, out_c, stride)]
        for _ in range(1, n):
            layers.append(GhostBottle(out_c, mid_c, out_c, 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x):
        """Return three feature maps [C3, C4, C5] for the FPN neck."""
        x = self.stem(x)
        x = self.stage1(x)
        c3 = self.stage2(x)               # [B, 128, H/8,  W/8]
        c4 = self.rif(self.stage3(c3))    # [B, 192, H/16, W/16]
        x = self.stage4(c4)
        x = self.strip(x)
        c5 = self.cbam(x)                 # [B, 256, H/32, W/32]
        return [c3, c4, c5]

    def forward(self, x):
        feats = self.forward_features(x)
        x = self.pool(feats[-1])
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)
        return x
