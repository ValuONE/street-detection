from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .head import HeadOutput, dist2bbox, make_anchors
from .loss_utils import bbox2dist, bbox_ciou, xywh2xyxy
from .tal import TaskAlignedAssigner


@dataclass
class LossGains:
    """Default loss configuration, used when no trainer supplies one.

    Attributes:
        box: Gain on the CIoU term.
        cls: Gain on the classification term.
        dfl: Gain on the L1 term.
    """

    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    epochs: int = 100


class BboxLoss(nn.Module):

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(loss_iou, loss_l1)``.

        Both terms are weighted by the anchor's target score, so well-aligned anchors
        steer the box branch more than marginal ones.

        Args:
            pred_dist: ``(b, A, 4)`` raw predicted ``(l, t, r, b)`` distances, in
                feature-map units. What the L1 term scores.
            pred_bboxes: ``(b, A, 4)`` the same predictions already decoded to
                ``(x1, y1, x2, y2)`` corners. What the CIoU term scores.
            anchor_points: ``(A, 2)`` anchor centres in feature-map units, needed to turn
                the target corners back into distances.
            target_bboxes: ``(b, A, 4)`` the assigner's box for each anchor, corners in
                feature-map units.
            target_scores: ``(b, A, nc)`` soft one-hot labels. The alignment metric sits
                in the true class column. Summed over classes it is the per-anchor weight.
            target_scores_sum: scalar, the batch's total alignment mass. The shared
                normalizer for all three loss terms.
            fg_mask: ``(b, A)`` bool, which anchors the assigner made positive. Background
                anchors have no box to regress and are excluded entirely.
            imgsz: ``(2,)`` input resolution as ``(h, w)`` in pixels, recovered from the
                finest feature map. Normalizes the L1 term to fractions of the image.
            stride: ``(A, 1)`` pixels per feature-map unit for each anchor. Converts the
                L1 term's two sides to a common pixel scale before that normalization.
        """
        # Per-anchor weight
        weight = target_scores[fg_mask].sum(-1, keepdim=True)

        # Intersection over Union
        iou = bbox_ciou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        # compute loss
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # make the loss independent of the level (P3, P4, P5)
        target_ltrb = bbox2dist(anchor_points, target_bboxes) * stride # corners -> ltrb, then to px
        target_ltrb[..., 0::2] /= imgsz[1]  # l, r  <- width
        target_ltrb[..., 1::2] /= imgsz[0]  # t, b  <- height

        # same for the predictions
        pred_dist = pred_dist * stride
        pred_dist[..., 0::2] /= imgsz[1]
        pred_dist[..., 1::2] /= imgsz[0]

        # weighted l1 loss (absolute distance each edge is off) per anchor
        loss_l1 = F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight

        # overall l1 loss
        loss_l1 = loss_l1.sum() / target_scores_sum

        return loss_iou, loss_l1


class DetectionLoss:
    """
    Loss for one detection branch.
    """

    def __init__(self, model: nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):

        device = next(model.parameters()).device
        m = model.detect

        # Sigmoid(logits) with binary cross entropy loss
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = model.args  # box / cls / dfl gains and the epoch count
        self.stride = m.stride
        self.nc = m.nc
        self.device = device
        self.loss_names = ("box_loss", "cls_loss", "l1_loss")

        # alpha=0.5 / beta=6.0: the alignment metric is `score^alpha * iou^beta`,
        # therefore IoU dominates the choice of positive anchor
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = BboxLoss().to(device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            targets: ``(n_objects_in_batch, 5)`` rows of ``(batch_idx, cls, x, y, w, h)``
                with the box normalized to 0-1.
            batch_size: Images in the batch. Needed explicitly because an image with no
                objects contributes no rows and would otherwise be invisible.
            scale_tensor: ``(w, h, w, h)`` pixel dimensions to multiply the normalized box by.

        Returns:
            ``(batch_size, max_objects, 5)`` of ``(cls, x1, y1, x2, y2)`` in pixels, zero padded.
        """
        nl, ne = targets.shape
        if nl == 0:
            # Handle no objects in the batch case
            return torch.zeros(batch_size, 0, ne - 1, device=self.device)

        batch_idx = targets[:, 0].long()

        # Width of the output is the largest object count of any single image in the batch.
        # other images are padded
        _, counts = batch_idx.unique(return_counts=True)
        out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)

        # Calculating object index within the own image by building cumulative histogram
        # and subtracting the count of objects before the target image.
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
        offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
        offsets = offsets.cumsum(0)
        within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]

        # row k of targets lands at out[batch_idx[k], within_idx[k]]
        out[batch_idx, within_idx] = targets[:, 1:]

        # Columns 1:5 are the box, mul_ scales normalised xywh to pixel xywh
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def get_assigned_targets_and_loss(self, preds: HeadOutput, batch: dict[str, Any]) -> tuple:
        """Assign ground truth to anchors, then compute the three loss terms.

        Args:
            preds: one branch's output from ``Detect.forward_head``.
            batch: the dataloader's batch.

        Returns:
            A 3-tuple of ``(assignment_artifacts, loss, loss_items)``. The first element
            exists so subclasses can reuse the assignment for mask or keypoint terms;
            nothing in this package reads it, and ``loss()`` discards it.
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, l1

        # (b, C, A) to (b, A, C)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()  # (b, 8400, 4)
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()  # (b, 8400, nc)

        # build anchor grid and corresponding stride tensor
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]

        # Recover input resolution from the finest feature map: 80 * 8 = 640
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0] # (h, w)

        # Flatten the label dict into (n, 6), with each row being (batch_idx, cls, x, y, w, h)
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        # imgsz is (h, w), box is (x, y, x, y) -> [1, 0, 1, 0] reorder.
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # (b, n, 1) and (b, n, 4)

        # A padded slot is all zeros ->coordinates sum to 0
        # gt_bboxes: (b, n_max, 4)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0) # greater_than -> 1.0 / 0.0 mask

        # decode predictions, in feature-map units.
        pred_bboxes = dist2bbox(pred_distri, anchor_points, dim=-1)  # feature-map units

        # assigner decides which anchor is responsible
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # shared normalizer
        target_scores_sum = max(target_scores.sum(), 1)

        # cls (binary cross entropy) loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():  # skip when the batch has no assigned anchors
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                # assigner returns pixels, but BboxLoss works in feature-map units
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        # applying gains -> weighting the different loss types
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl  # l1, however has the old name

        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )

    def loss(self, preds: HeadOutput, batch: dict[str, torch.Tensor]):
        """
        Scale by batch size so the gradient magnitude does not depend on it.
        """
        batch_size = preds["boxes"].shape[0]
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        return loss * batch_size, loss_detach

    def __call__(self, preds, batch: dict[str, torch.Tensor]):
        return self.loss(self.parse_output(preds), batch)

    @staticmethod
    def parse_output(preds):
        """Accept either the training dict or the ``(det, preds)`` eval tuple."""
        return preds[1] if isinstance(preds, tuple) else preds


class E2ELoss:
    """
    Combine the one-to-many and one-to-one losses with a shifting gain.
    This way the models learns quickly at first and trains later on for the inference
    relevant task.
    """

    def __init__(self, model: nn.Module):
        # tal_topk = 10 allows multiple anchors
        self.one2many = DetectionLoss(model, tal_topk=10)
        # tal_topk2 = 1 enforces only one anchor per object -> no NMS needed
        self.one2one = DetectionLoss(model, tal_topk=7, tal_topk2=1)

        self.updates = 0        # epochs elapsed, advanced by update()
        self.total = 1.0        # the sum of both gains
        self.o2m = 0.8          # current one2many gain
        self.o2o = self.total - self.o2m
        self.o2m_copy = self.o2m    # the starting value
        self.final_o2m = 0.1        # last epoch o2m value

    def __call__(self, preds, batch: dict[str, torch.Tensor]):
        """
        Score both branches and apply current gain.

        Args:
            preds: The head's training dict, or the eval tuple
            batch: The dataloader batch passed to both branches

        Returns:
            (loss, loss_items)
        """
        preds = DetectionLoss.parse_output(preds)
        loss_one2many = self.one2many.loss(preds["one2many"], batch)
        loss_one2one = self.one2one.loss(preds["one2one"], batch)
        return loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o, loss_one2one[1]

    def update(self) -> None:
        """
        Advance the schedule by one epoch.
        """
        self.updates += 1
        self.o2m = self.decay(self.updates)
        self.o2o = max(self.total - self.o2m, 0)

    def decay(self, x: int) -> float:
        """Linear ramp of the one2many gain from ``o2m_copy`` (0.8) down to ``final_o2m``.

        Args:
            x: Epochs elapsed.

        Returns:
            The one2many gain for that epoch.
        """
        return (max(1 - x / max(self.one2one.hyp.epochs - 1, 1), 0)
                * (self.o2m_copy - self.final_o2m) + self.final_o2m)


def build_criterion(model: nn.Module, args: Any = None) -> E2ELoss:
    """The loss YOLO26 trains with.

    Args:
        model: The model to train.
        args: Loss configuration.

    Returns:
        An :class:`E2ELoss` over both head branches.
    """
    if args is None:
        args = getattr(model, "args", None) or LossGains()
    model.args = args
    return E2ELoss(model)
