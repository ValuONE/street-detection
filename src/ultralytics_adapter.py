"""
Providing a wrapper around the self implemented model to run it via the ultralytics api.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from ultralytics.models import yolo
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

from .loss import build_criterion
from .model import YOLO26


class MyDetectionModel(DetectionModel):
    """
    Wrapping the detection model interface around the won implementation.

    Args:
        cfg: Ignored.
        ch: Input channels.
        nc: Number of classes.

    Attributes:
        net: The model.
        model: Sequential(net, net.detect), existing so self.model[-1] is the head.
        save: Empty, because it's unused.
    """

    def __init__(self, cfg: Any = None, ch: int = 3, nc: int = 6, verbose: bool = False):
        nn.Module.__init__(self)
        net = YOLO26(nc=nc, ch=ch)
        self.net = net
        self.model = nn.Sequential(net, net.detect)
        self.save: list[int] = []
        self.yaml = {"nc": nc, "channels": ch, "scale": "n", "inplace": True}
        self.names = {i: str(i) for i in range(nc)}
        self.inplace = True
        self.nc = nc
        self.stride = net.detect.stride
        # Read by the validator to skip NMS, and by set_head_attr. The scratch Detect is
        # always end-to-end.
        net.detect.end2end = True
        net.detect.agnostic_nms = False  # never read; silences a set_head_attr warning
        self.criterion = None

    def predict(self, x: torch.Tensor, profile: bool = False, augment: bool = False, embed: Any = None):
        """.
        Args:
            x: ``(B, 3, H, W)`` normalized images.
            profile, augment, embed: Only for signature compatibility.

        Returns:
            Whatever YOLO26.forward() returns.
        """
        return self.net(x)

    def init_criterion(self):
        """
        Instantiating the loss with the engine's cfg.

        Passing args explicitly matters: without it the loss falls back to LossGains()
        (box=7.5, epochs=100) and silently ignores what was passed to train(). The epoch
        count is the damaging one -- it drives the one2many -> one2one gain schedule, so a
        wrong value leaves the deployed head as the minor objective for the whole run.

        ``build_criterion`` reads the gains off ``model.args``, so the engine's cfg is
        assigned there first -- it carries box/cls/dfl and epochs under the same names.
        """
        args = getattr(self, "args", None)
        if args is not None:
            self.net.args = args
        return build_criterion(self.net)

    def loss(self, batch: dict[str, Any], preds: Any = None):
        """
        Compute the loss for one batch.
        Mirrors ``BaseModel.loss`` exactly, but the own predict() method is used.
        """
        if self.criterion is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.predict(batch["img"])
        return self.criterion(preds, batch)

    def fuse(self, verbose: bool = True, imgsz: int = 640):
        """
        No-op to disable folding the BatchNorm into the preceding conv.
        """
        return self

    def _apply(self, fn):
        """
        Move the head's plain tensors on .to() / .cuda() / .half(), which would normally be done
        by BaseModel._apply.
        """
        nn.Module._apply(self, fn)
        head = self.net.detect
        head.stride = fn(head.stride)
        head.anchors = fn(head.anchors)
        head.strides = fn(head.strides)
        self.stride = head.stride  # the engine reads stride off the model, not the head
        return self


class MyTrainer(DetectionTrainer):
    """DetectionTrainer with get_model() overwritten."""

    def get_model(self, cfg: Any = None, weights: Any = None, verbose: bool = True):
        """Build the scratch model instead of parsing a YAML.

        Args:
            cfg: Ignored.
            weights: Optional state to load.
            verbose: Ignored.

        Returns:
            A MyDetectionModel with the dataset's class names.
        """
        model = MyDetectionModel(nc=self.data["nc"], ch=self.data["channels"], verbose=verbose)
        model.names = self.data["names"]
        if weights:
            model.load(weights)
        return model


class MyYOLO(yolo.model.YOLO):
    """
    YOLO wrapper around the own model.
    """

    def __init__(self, model: Any = "yolo26n.yaml", task: str = "detect", verbose: bool = False):
        """
        Args:
            model: A checkpoint written by this adapter, or the default placeholder. The
                placeholder's contents are never used, but are needed due to the engines
                path checking.
            task: Only ``"detect"`` is overridden; other tasks fall through to stock.
            verbose: Passed to the base class.
        """
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Stock map with the detect entry's model and trainer replaced."""
        m = super().task_map
        m["detect"] = {
            "model": MyDetectionModel,
            "trainer": MyTrainer,
            "validator": yolo.detect.DetectionValidator,
            "predictor": yolo.detect.DetectionPredictor,
        }
        return m
