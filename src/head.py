from __future__ import annotations

import copy
import math
from typing import TypedDict

import torch
from torch import nn

from .blocks import Conv


class HeadOutput(TypedDict):
    """What one head branch emits:
        Attributes:
        boxes: ``(B, 4, A)`` predicted ``(l, t, r, b)`` distances, in stride units.
        scores: ``(B, nc, A)`` raw class logits, without sigmoid.
        feats: The three source feature maps, passed through so downstream code can
            rebuild the anchor grid from their spatial shapes.
    """

    boxes: torch.Tensor
    scores: torch.Tensor
    feats: list[torch.Tensor]


def make_anchors(
    feats: list[torch.Tensor], strides: torch.Tensor, grid_cell_offset: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:

    """Build the anchor point grid for every pyramid level.

    80x80 + 40x40 + 20x20 = 8400 points at 640x640

    Args:
        feats: The P3/P4/P5 feature maps
        strides: The stride of each level
        grid_cell_offset: default 0.5 (cell center).

    Returns:
        ``anchor_points`` ``(8400, 2)`` in feature-map units
        ``stride_tensor`` ``(8400, 1)`` pixels-per-unit.
        Multiplying a box by the ``stride_tensor`` converts
        it to input-image pixels.
    """
    anchor_points, stride_tensor = [], []

    dtype, device = feats[0].dtype, feats[0].device
    for i, feat in enumerate(feats):
        h, w = feat.shape[2:]
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset  # 0.5, 1.5, ... w-0.5
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset
        # indexing="ij" -> first axis y and the second x or (H, W).
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        # (h, w, 2) of (x, y)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        # One value per point as a column, allows broadcasts against (A, 4) boxes.
        stride_tensor.append(torch.full((h * w, 1), float(strides[i]), dtype=dtype, device=device))
    # fine to coarse: 6400 points (P3) + 1600 (P4), 400 P5.
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Turn distances (left, top, right, bottom) into boxes (x1, y1, x2, y2).

    Args:
        distance: ``(B, 4, A)`` distances as ``(l, t, r, b)`` in stride units.
        anchor_points: ``(1, 2, A)`` or any broadcastable shape, in stride units.
        dim: Axis holding the 4 distances. In this case 1.

    Returns:
        ``(B, 4, A)`` boxes as ``(x1, y1, x2, y2)``, in feature-map units.
        Multiplying with the stride gets the values in pixels
    """
    lt, rb = distance.chunk(2, dim)
    # Subtract for up-left, add for down-right
    return torch.cat((anchor_points - lt, anchor_points + rb), dim)


class Detect(nn.Module):
    """detection head over the P3/P4/P5 feature maps.

    Each level has two independent branches:

    * ``cv2``: box regression, two dense 3x3 convs into a 1x1 emitting 4 channels (l. t. r ,b).
    * ``cv3``: classification, two depthwise-separable stacks into a 1x1 emitting ``nc`` channels.

    ``one2one_cv2``/``one2one_cv3`` are independent copies trained with one-to-one assignment.
    For inference the o2o branch is used.

    Args:
        nc: Number of classes.
        ch: Input channels of the levels, in this case``(64, 128, 256)``.

    Attributes:
        nl: Number of pyramid levels, ``len(ch)``.
        no: Channels per anchor, ``nc + 4``, classes + l, t, r, b.
        stride: Pixels per feature-map unit per level.
        shape: Cached input shape guarding the anchor-grid rebuild.
        anchors, strides: The cached grid from ``make_anchors`` as
            ``(2, A)`` and ``(1, A)`` (broadcastable against ``(B, 4, A)`` boxes).
    """

    max_det = 300  # fixed number of detections returned per image

    def __init__(self, nc: int = 80, ch: tuple[int, ...] = ()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)  # number of pyramid levels
        self.no = nc + 4  # channels per anchor
        self.stride = torch.zeros(self.nl)  # filled in by the model's dummy forward
        self.shape = None  # cached feature shape, guards the anchor grid rebuild
        self.anchors = torch.empty(0)
        self.strides = torch.empty(0)

        # intermediate channel sizes
        c2 = max(16, ch[0] // 4, 4)  # box branch width
        c3 = max(ch[0], min(nc, 100))  # cls branch width

        # dense/full box branch:
        # Conv2d as last layer, because of the regressional characteristic of the value
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4, 1)) for x in ch
        )
        # Classification branch:
        # 3x3 stack with g=channels (no channel mixing only spatial)
        # and a final 1x1 layer (only channel mixing no spatial).
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),      # depthwise at input width
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),  # depthwise at branch width
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        # Deep copies, because of the different trainings assignment
        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)

    @property
    def one2many(self) -> dict[str, nn.Module]:
        return {"box_head": self.cv2, "cls_head": self.cv3}

    @property
    def one2one(self) -> dict[str, nn.Module]:
        return {"box_head": self.one2one_cv2, "cls_head": self.one2one_cv3}

    def forward_head(self, x: list[torch.Tensor], box_head: nn.Module, cls_head: nn.Module) -> HeadOutput:
        """
        Run one branch and flatten the three pyramid levels into one  anchor axis:
        80x80 + 40x40 + 20x20 = 8400 anchors
        This makes the anchors independent of the level, however the concatenation order
        must always be identical (fine to coarse P3 -> P5)

        Args:
            x: The three feature maps, ``[(B, ch[i], H_i, W_i)]``, fine to coarse.
            box_head: Per-level ``nn.ModuleList`` emitting 4 channels.
            cls_head: Per-level ``nn.ModuleList`` emitting ``nc`` channels.

        Returns:
            ``{"boxes": (B, 4, A),
               "scores": (B, nc, A),
               "feats": x}``
               Scores are raw logits.
        """
        bs = x[0].shape[0]

        boxes = torch.cat([box_head[i](x[i]).view(bs, 4, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)

        return {"boxes": boxes, "scores": scores, "feats": x}

    def forward(self, x: list[torch.Tensor]):
        """
        In training the one2one branch sees detached features. Both branches would
        otherwise push conflicting gradients into the same backbone.

        Args:
            x: The three feature maps from the neck, fine to coarse.

        Returns:
            Training: the ``{"one2many": ..., "one2one": ...}`` dict of raw outputs.
            Eval: a ``(detections, preds)`` pair, ``(B, max_det, 6)`` with (x1, y1, x2, y2, conf, cls)
                    decoded from the one2one branch,
                    and the raw ``preds`` dict as in training.
        """
        preds = {
            # sees the live features: backbone is trained using this branch.
            "one2many": self.forward_head(x, **self.one2many),
            # sees detached features
            "one2one": self.forward_head([xi.detach() for xi in x] if self.training else x, **self.one2one),
        }
        if self.training:
            return preds  # raw dicts

        y = self._inference(preds["one2one"])
        # Permute to anchor-major for postprocessing
        return self.postprocess(y.permute(0, 2, 1)), preds

    def _inference(self, x: HeadOutput) -> torch.Tensor:
        """Decode boxes to pixels and applies the sigmoid function to the class logits.

        Sigmoid instead of softmax, so the sum doesn't have to sum up to 1 and therefore
        an anchor can have no class.

        Args:
            x: output dict from ``forward_head``.

        Returns:
            ``(B, 4 + nc, A)`` boxes in pixel and probabilities (x1, y1, x2, y2, nc...).
        """
        # (B, 4, A) for the boxes and (B, nc, A) for classes
        # Concat along the second/first dimension
        return torch.cat((self._decode_boxes(x), x["scores"].sigmoid()), 1)

    def _decode_boxes(self, x: HeadOutput) -> torch.Tensor:
        """Convert predicted distances into pixel-space xyxy corners.

        Args:
            x: One branch's output dict; ``x["boxes"]`` and ``x["feats"]`` are read.

        Returns:
            ``(B, 4, A)`` boxes as ``(x1, y1, x2, y2)`` in input-image pixels. Values may
            fall outside the image -- nothing clips them here.
        """
        shape = x["feats"][0].shape
        # rebuild the anchor grid only when the input size changes
        if self.shape != shape:
            # Transposing make_anchors' (A, 2) and (A, 1) to (2, A) and (1, A),
            # puts the coordinate axis where the boxes keep theirs
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape
        # getting pixels by multiplying each anchor by its own level's stride
        return dist2bbox(x["boxes"], self.anchors.unsqueeze(0)) * self.strides

    def bias_init(self) -> None:
        """
        Set the output biases carefully so that the first training steps are not wasted.

        """
        for branch in (self.one2many, self.one2one):
            for i, (box, cls) in enumerate(zip(branch["box_head"], branch["cls_head"])):
                # [-1] is nn.Conv2d, the only one with a bias, because the others have a BatchNorm.
                # setting 2.0 for positive distances
                box[-1].bias.data[:] = 2.0
                # log(objects / classes / anchors): the log-odds of a class firing at a
                # random anchor. The assumption of 5 objects per level is used.
                # (640 / stride)^2 is that level's anchor count.
                cls[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Replacement of NMS with top-k.

        Args:
            preds: as ``(B, A, 4 + nc)``

        Returns:
            ``(B, max_det, 6)`` rows of ``[x1, y1, x2, y2, conf, cls]``, confidence descending.
        """
        boxes, scores = preds.split([4, self.nc], dim=-1)
        scores, classes, idx = self.get_topk_index(scores, self.max_det)
        # Expanding (B, k, 1) to (B, k, 4) gathers all four coordinates of each winning
        # anchor, and therefore sorting it
        boxes = boxes.gather(dim=1, index=idx.expand(-1, -1, 4))
        # (B, k, 4) + (B, k, 1) + (B, k, 1) -> (B, k, 6)
        return torch.cat([boxes, scores, classes], dim=-1)

    def get_topk_index(self, scores: torch.Tensor, max_det: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Two-stage top-k over the (anchor, class) score matrix.

        Args:
            scores: ``(B, A, nc)`` per-anchor class probabilities, post-sigmoid.
            max_det: Slots to return, clamped to ``A``.

        Returns:
            ``scores`` ``(B, k, 1)`` confidences,
            ``classes`` ``(B, k, 1)`` class indices as floats,
            ``idx`` ``(B, k, 1)`` indices into the anchor axis, ready for``Tensor.gather``.
        """
        _, anchors, nc = scores.shape
        # Handle feature map smaller than max_det
        k = min(max_det, anchors)

        # Stage 1: Rank anchors by their single best class and keep the k best.
        # [1] is topk's indices, not its values.
        # (B, k, 1) best anchors
        ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)

        # Filter scores using the index
        # (B, k, nc)
        scores = scores.gather(dim=1, index=ori_index.expand(-1, -1, nc))

        # Stage 2: Rank every surviving (anchor, class) pair.
        # One anchor may take several slots
        scores, index = scores.flatten(1).topk(k)

        # index addresses the flattened (k, nc) matrix
        # -> // nc is the row, % nc the class.
        # filter original index
        idx = ori_index.gather(dim=1, index=(index // nc).unsqueeze(-1))

        # [..., None] restores the axis topk consumed, so all three return as (B, k, 1).
        # (index % nc)[..., None] do unflatten the index into classes
        return scores[..., None], (index % nc)[..., None].float(), idx
