"""Rotated anchor generation.

Generates ``len(scales) * len(ratios) * len(angles)`` anchors per spatial
location, each as ``(cx, cy, w, h, angle)`` with angle in radians.

With an FPN, each level handles a single object scale, so ``level_scales`` gives
the scales per level (e.g. ``[[32], [64], [128]]``) rather than every scale at
every level. Angles are passed in DEGREES and converted to radians here.
"""

import math

import torch


def generate_rotated_anchors(feature_sizes, strides, level_scales, anchor_ratios,
                             anchor_angles, device):
    """
    Args:
        feature_sizes: list of (H, W) per FPN level.
        strides:       list of stride per level.
        level_scales:  list of scale lists, one per level, e.g. [[32],[64],[128]].
        anchor_ratios: e.g. [0.5, 1.0, 2.0].
        anchor_angles: angles in DEGREES (converted to radians internally).
    Returns:
        list of ``[H*W*A, 5]`` anchor tensors, one per level. ``A`` is
        ``len(level_scales[idx]) * len(anchor_ratios) * len(anchor_angles)``.
    """
    anchors_per_level = []
    for idx, (H, W) in enumerate(feature_sizes):
        stride = strides[idx]
        anchors = []
        for i in range(H):
            for j in range(W):
                cx = (j + 0.5) * stride
                cy = (i + 0.5) * stride
                for scale in level_scales[idx]:
                    for ratio in anchor_ratios:
                        w = scale * math.sqrt(ratio)
                        h = scale / math.sqrt(ratio)
                        for angle in anchor_angles:
                            anchors.append([cx, cy, w, h, math.radians(angle)])
        anchors_per_level.append(torch.tensor(anchors, dtype=torch.float32, device=device))
    return anchors_per_level
