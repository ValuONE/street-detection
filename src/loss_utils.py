from __future__ import annotations

import math

import torch


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """(cx, cy, w, h) -> (x1, y1, x2, y2)."""
    y = torch.empty_like(x)
    xy, wh = x[..., :2], x[..., 2:] / 2
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
    return y


def xyxy2xywh(x: torch.Tensor) -> torch.Tensor:
    """(x1, y1, x2, y2) -> (cx, cy, w, h)."""
    y = torch.empty_like(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Complete IoU between two sets of xyxy boxes.


    For non overlapping boxes is the plain IoU zero and therefore gradient free.
    The following two terms however still add infomation:

    * ``rho2 / c2``: normalized distance between centers -> direction to move
    * ``v * alpha``: disagreement in aspect ratio, weighted so it only matters once the
      boxes already overlap reasonably.

    Returns values in (-1, 1].
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # smallest enclosing box, width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # ... and height
    c2 = cw.pow(2) + ch.pow(2) + eps  # its diagonal, squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
    v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
    with torch.no_grad():  # alpha is a weight
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
    """
    (x1, y1, x2, y2) -> (left, top, right, bottom) distances from each anchor.
    """
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
