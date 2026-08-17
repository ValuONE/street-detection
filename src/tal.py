from __future__ import annotations

import torch
from torch import nn

from .loss_utils import bbox_ciou, xywh2xyxy, xyxy2xywh


class TaskAlignedAssigner(nn.Module):
    """Assign ground-truth objects to anchors by a combined classification/IoU metric.

    Args:
        topk: number of candidate anchors per object
        num_classes: number of classes
        alpha, beta: parameters for the alignment metric
        stride: the three pyramid strides
        eps: division guard.
        topk2: number of anchors per object
    """

    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        stride: list[float] | None = None,
        eps: float = 1e-9,
        topk2: int | None = None,
    ):
        super().__init__()
        self.topk = topk
        self.topk2 = topk2 or topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.stride = stride if stride is not None else [8, 16, 32]
        self.stride_val = self.stride[1] if len(self.stride) > 1 else self.stride[0]
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the assignment.

        Args:
            pd_scores: (bs, num_anchors, nc) predicted class probabilities
            pd_bboxes: (bs, num_anchors, 4) predicted boxes, xyxy in pixels
            anc_points: (num_anchors, 2) anchor centres in pixels
            gt_labels: (bs, n_max_boxes, 1) ground-truth classes
            gt_bboxes: (bs, n_max_boxes, 4) ground-truth boxes, xyxy in pixels
            mask_gt: (bs, n_max_boxes, 1) ground-truth mask

        Returns:
            target_labels (bs, num_anchors),
            target_bboxes (bs, num_anchors, 4),
            target_scores (bs, num_anchors, nc),
            fg_mask (bs, num_anchors),
            target_gt_idx (bs, num_anchors)
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]

        # all-background batch: every anchor is negative
        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores,
            pd_bboxes,
            gt_labels,
            gt_bboxes,
            anc_points,
            mask_gt
        )
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos,
            overlaps,
            self.n_max_boxes,
            align_metric
        )
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels,
            gt_bboxes,
            target_gt_idx,
            fg_mask
        )

        # apply mask
        align_metric *= mask_pos
        # get best anchors regarding the alignment metric
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        # get best anchors regarding the IoU
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        # normalization (align_metric / pos_align_metrics)
        # rescaling (* pos_overlaps) to best IoU
        # max over object axis -> only one object per anchor
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        # apply mask
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        # filter anchors being inside the ground truth box
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        # calculate metrics
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # mask for topk
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        return mask_topk * mask_in_gts * mask_gt, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """
        Compute ``score^alpha * iou^beta`` for every (object, anchor) pair.
        """
        # number of anchors
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        batch_ind = torch.arange(self.bs, device=pd_scores.device)[:, None]
        # for each object get every anchor's score for that object's class.
        bbox_scores[mask_gt] = pd_scores[batch_ind, :, gt_labels.squeeze(-1).long()][mask_gt]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        return bbox_scores.pow(self.alpha) * overlaps.pow(self.beta), overlaps

    def iou_calculation(self, gt_bboxes: torch.Tensor, pd_bboxes: torch.Tensor) -> torch.Tensor:
        """
        CIoU, clamped non-negative. This is needed due to the (possible) even exponent in the metric.
        Otherwise -1 (no overlap) would be equal to 1 (identical boxes)
        """
        return bbox_ciou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics: torch.Tensor, topk_mask: torch.Tensor | None = None) -> torch.Tensor:
        """keep each object's topk highest-metric anchors"""
        # (b, n, topk) best anchors per object + indices into the anchor axis
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        # fallback for no mask
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        topk_idxs.masked_fill_(~topk_mask, 0)  # padded objects collapse onto anchor 0

        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8))
        count_tensor.masked_fill_(count_tensor > 1, 0)
        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """Gather the assigned object's label and box for every anchor.

        ``target_gt_idx`` indexes objects within an image -> offset by
        ``batch * n_max_boxes`` to index the flattened batch
        """
        # per-image object index -> index into flattened (b * n_max) object list
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        target_labels.clamp_(0)
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int8,
            device=target_labels.device,
        )
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        # background anchors get all-zeros
        target_scores = target_scores * (fg_mask[:, :, None] > 0)

        return target_labels, target_bboxes, target_scores

    def select_candidates_in_gts(
        self, xy_centers: torch.Tensor, gt_bboxes: torch.Tensor, mask_gt: torch.Tensor, eps: float = 1e-9
    ) -> torch.Tensor:
        """
        Selects anchor as candidate only if its center is within the box
        """
        gt_bboxes_xywh = xyxy2xywh(gt_bboxes)
        # boxes thinner than the finest stride could contain no anchor center at all
        wh_mask = gt_bboxes_xywh[..., 2:] < self.stride[0]
        gt_bboxes_xywh[..., 2:] = torch.where(
            (wh_mask * mask_gt).bool(),
            torch.tensor(self.stride_val, dtype=gt_bboxes_xywh.dtype, device=gt_bboxes_xywh.device),
            gt_bboxes_xywh[..., 2:],
        )
        gt_bboxes = xywh2xyxy(gt_bboxes_xywh)

        # (b, n, 1, 2) each
        lt, rb = gt_bboxes.unsqueeze(2).chunk(2, 3)
        return ((xy_centers - lt > eps) & (rb - xy_centers > eps)).all(3)

    def select_highest_overlaps(
        self, mask_pos: torch.Tensor, overlaps: torch.Tensor, n_max_boxes: int, align_metric: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Resolve multiple objects per anchor and select topk2
        """
        # (b, A) number of objects claiming each anchor >1 -> conflict to resolve.
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
            fg_mask = mask_pos.sum(-2)

        # one2one head only (topk2 == 1): for each object only its single best anchor
        if self.topk2 != self.topk:
            align_metric = align_metric * mask_pos
            topk_align_idx = torch.topk(align_metric, self.topk2, dim=-1, largest=True).indices
            topk_idx = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            topk_idx.scatter_(-1, topk_align_idx, 1.0)
            mask_pos *= topk_idx
            fg_mask = mask_pos.sum(-2)

        target_gt_idx = mask_pos.argmax(-2)
        return target_gt_idx, fg_mask, mask_pos
