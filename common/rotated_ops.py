"""Rotated-box geometry utilities.

Boxes are ``(cx, cy, w, h, theta)`` with ``theta`` in radians. ``box_iou_rotated``
uses ``mmcv.ops.box_iou_rotated`` (CUDA/CPU). mmcv is required -- the previous
shapely CPU fallback was removed because it is far too slow for training-scale use.
"""

import torch

try:
    from mmcv.ops import box_iou_rotated as _mmcv_box_iou_rotated
    _HAS_MMCV = True
except Exception:  # pragma: no cover - depends on environment
    _HAS_MMCV = False

_MMCV_REQUIRED = (
    "mmcv is required for rotated IoU/NMS. Install it on the training machine, e.g.\n"
    "  pip install -U openmim && mim install mmcv\n"
    "(or `pip install mmcv` matching your torch/CUDA)."
)


def get_rotated_corners(boxes):
    """Convert ``(cx, cy, w, h, angle)`` boxes to their 4 corner points.

    Args:
        boxes: ``[N, 5]`` tensor, angle in radians.
    Returns:
        ``[N, 4, 2]`` corner coordinates (x, y).
    """
    cx, cy, w, h, angle = boxes.unbind(-1)
    device = boxes.device
    corners = torch.tensor(
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
        device=device, dtype=boxes.dtype,
    )  # [4, 2]
    corners = corners[None, :, :] * torch.stack([w, h], dim=-1)[:, None, :]
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    R = torch.stack(
        [torch.stack([cos_a, -sin_a], dim=-1), torch.stack([sin_a, cos_a], dim=-1)],
        dim=-2,
    )  # [N, 2, 2]
    rotated = torch.einsum("nij,nkj->nki", R, corners)
    rotated += torch.stack([cx, cy], dim=-1)[:, None, :]
    return rotated  # [N, 4, 2]


def box_iou_rotated(boxes1, boxes2):
    """Pairwise rotated IoU between two sets of ``(cx, cy, w, h, theta)`` boxes.

    Args:
        boxes1: ``[N, 5]`` (angle in radians).
        boxes2: ``[M, 5]`` (angle in radians).
    Returns:
        ``[N, M]`` IoU matrix on ``boxes1.device``.
    """
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)
    if not _HAS_MMCV:
        raise ImportError(_MMCV_REQUIRED)
    # mmcv expects float32 contiguous tensors; aligned=False -> pairwise NxM.
    return _mmcv_box_iou_rotated(boxes1.float(), boxes2.float(), aligned=False)
