"""Rotated-box geometry utilities.

Boxes are ``(cx, cy, w, h, theta)`` with ``theta`` in radians. ``box_iou_rotated``
computes pairwise IoU using ``mmcv.ops.box_iou_rotated`` when available (CUDA/CPU),
otherwise a ``shapely`` polygon-intersection fallback.
"""

import torch

try:  # Preferred: standard, fast, field-standard op.
    from mmcv.ops import box_iou_rotated as _mmcv_box_iou_rotated
    _HAS_MMCV = True
except Exception:  # pragma: no cover - depends on environment
    _HAS_MMCV = False

try:
    from shapely.geometry import Polygon as _ShapelyPolygon
    _HAS_SHAPELY = True
except Exception:  # pragma: no cover - depends on environment
    _HAS_SHAPELY = False


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


def _box_iou_rotated_shapely(boxes1, boxes2):
    """CPU fallback rotated IoU via shapely. Returns ``[N, M]`` on ``boxes1.device``."""
    device = boxes1.device
    n, m = boxes1.shape[0], boxes2.shape[0]
    ious = torch.zeros((n, m), dtype=torch.float32)
    if n == 0 or m == 0:
        return ious.to(device)

    c1 = get_rotated_corners(boxes1.float().cpu()).numpy()
    c2 = get_rotated_corners(boxes2.float().cpu()).numpy()
    polys1 = [_ShapelyPolygon(c) for c in c1]
    polys2 = [_ShapelyPolygon(c) for c in c2]
    areas2 = [p.area for p in polys2]

    for i, p1 in enumerate(polys1):
        a1 = p1.area
        for j, p2 in enumerate(polys2):
            if not p1.is_valid or not p2.is_valid:
                continue
            inter = p1.intersection(p2).area
            union = a1 + areas2[j] - inter
            if union > 0:
                ious[i, j] = inter / union
    return ious.to(device)


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

    if _HAS_MMCV:
        # mmcv expects float32 contiguous tensors; aligned=False -> pairwise NxM.
        return _mmcv_box_iou_rotated(boxes1.float(), boxes2.float(), aligned=False)
    if _HAS_SHAPELY:
        return _box_iou_rotated_shapely(boxes1, boxes2)
    raise ImportError(
        "Rotated IoU requires either `mmcv` (preferred, CUDA) or `shapely`.\n"
        "Install one of:  pip install mmcv  |  pip install shapely"
    )
