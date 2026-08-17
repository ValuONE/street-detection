from __future__ import annotations

import torch
from torch import nn

from .blocks import C2PSA, C3k2, Conv, SPPF
from .head import Detect

STREET_CLASSES = ("car", "truck", "bus", "person", "bicycle", "motorcycle")


class YOLO26(nn.Module):
    """
    Corresponds to ``cfg/models/26/yolo26.yaml``.
    Shapes for a 640x640 input::
        stem   -> 320x320x16
        down2  -> 160x160x32
        down3  ->  80x80x64 (P3 head)
        down4  ->  40x40x128 (P4 head)
        down5  ->  20x20x256 (P5 head)
    """

    def __init__(self, nc: int = len(STREET_CLASSES), ch: int = 3):
        super().__init__()
        self.nc = nc

        # Channel counts after width scaling;
        # 64, 128, 256, 512, 1024 at width 1.0, these are multiplied by 0.25 due to the
        # nano architecture version
        c1, c2, c3, c4, c5 = 16, 32, 64, 128, 256

        # Backbone
        self.stem = Conv(ch, c1, 3, 2)                              # 0  P1/2
        self.down2 = Conv(c1, c2, 3, 2)                             # 1  P2/4
        self.stage2 = C3k2(c2, c3, 1, c3k=False, e=0.25)               # 2
        self.down3 = Conv(c3, c3, 3, 2)                             # 3  P3/8
        self.stage3 = C3k2(c3, c4, 1, c3k=False, e=0.25)               # 4
        self.down4 = Conv(c4, c4, 3, 2)                             # 5  P4/16
        self.stage4 = C3k2(c4, c4, 1, c3k=True)                        # 6
        self.down5 = Conv(c4, c5, 3, 2)                             # 7  P5/32
        self.stage5 = C3k2(c5, c5, 1, c3k=True)                        # 8
        self.sppf = SPPF(c5, c5, k=5, n=3, shortcut=True)                 # 9
        self.psa = C2PSA(c5, 1)                                        # 10

        # Neck: Feature Pyramid Network (FPN, top-down) then Path Aggregation Network (PAN, bottom-up)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")       # 11, 14
        self.fpn_p4 = C3k2(c5 + c4, c4, 1, c3k=True)                   # 13
        self.fpn_p3 = C3k2(c4 + c4, c3, 1, c3k=True)                   # 16  -> P3 head
        self.pan_down4 = Conv(c3, c3, 3, 2)                         # 17
        self.pan_p4 = C3k2(c3 + c4, c4, 1, c3k=True)                   # 19  -> P4 head
        self.pan_down5 = Conv(c4, c4, 3, 2)                         # 20
        self.pan_p5 = C3k2(c4 + c5, c5, 1, c3k=True, e=0.5, attn=True) # 22  -> P5 head

        # Head
        self.detect = Detect(nc=nc, ch=(c3, c4, c5))                      # 23
        self.head_channels = (c3, c4, c5)

        self._init_strides()

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Backbone and neck. Returns the P3/P4/P5 maps the head consumes."""
        x = self.stem(x)
        x = self.stage2(self.down2(x))
        p3 = self.stage3(self.down3(x))
        p4 = self.stage4(self.down4(p3))
        p5 = self.stage5(self.down5(p4))

        p5 = self.psa(self.sppf(p5))

        t4 = self.fpn_p4(torch.cat([self.upsample(p5), p4], 1))
        t3 = self.fpn_p3(torch.cat([self.upsample(t4), p3], 1))

        b4 = self.pan_p4(torch.cat([self.pan_down4(t3), t4], 1))
        b5 = self.pan_p5(torch.cat([self.pan_down5(b4), p5], 1))

        return [t3, b4, b5]

    def forward(self, x: torch.Tensor):
        """
        During training: ``{"one2many": {...}, "one2one": {...}}``, each with:
        boxes: (B, 4, 8400) with (left, top, right, bottom) for each anchor,
        scores: (B, nc, 8400) with the sum over the classes not necessarily being 1,
        bcause not softmax but the sigmoid function is used.
        feats: the feature maps the head got.

        During eval: ``(det, preds)`` where ``det`` is (B, 300, 6) as
        (x1, y1, x2, y2, conf, cls) in input-image pixels, already deduplicated.
        """
        return self.detect(self.forward_features(x))

    @torch.no_grad()
    def _init_strides(self, imgsz: int = 256) -> None:
        """
        Measure the head's strides by running one dummy image through the network, therefore
        they adapt to the network rather than being hardcoded to [8, 16, 32] (for the current architecture).
        """
        was_training = self.training
        self.eval()
        dummy = torch.zeros(1, 3, imgsz, imgsz)
        self.detect.stride = torch.tensor([imgsz / f.shape[-2] for f in self.forward_features(dummy)])
        self.detect.bias_init()
        self.detect.shape = None
        self.train(was_training)
