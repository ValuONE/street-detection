"""This module implements the necessary building blocks for the YOLO26 architecture."""

from __future__ import annotations

import torch
from torch import nn

class Conv(nn.Module):
    """The convolution block every other block in this module is built from.

    Runs ``Conv2d -> BatchNorm2d -> SiLU``, with an optional activation.

    Args:
        c1: Input channels.
        c2: Output channels.
        k: Kernel size, either a single int or a ``(kernel_height, kernel_width)`` pair.
        s: Stride. ``1`` keeps the resolution, ``2`` halves it. A strided ``Conv`` is the
            only thing in this module that resizes a feature map.
        p: Padding. ``None`` means same input size and output size if ``s`` is 1.
        g: Groups. How many input channels each output channel sees. ``1`` is normal:
            every output sees every input. ``g == c1 == c2`` is depthwise: one filter per
            channel, no cross-channel mixing.
        act: ``True`` for SiLU, ``False`` for ``nn.Identity``.
    """

    # Constants used for the batch norm
    BN_EPS = 1e-3
    BN_MOMENTUM = 0.03

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int | tuple[int, int] = 1,
        s: int = 1,
        p: int | tuple[int, int] | None = None,
        g: int = 1,
        act: bool = True,
    ):
        super().__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else (k[0] // 2, k[1] // 2)

        # Configure components
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=self.BN_EPS, momentum=self.BN_MOMENTUM)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the configured components.

        Args:
            x: ``(B, c1, H, W)`` feature map.

        Returns:
            ``(B, c2, H // s, W // s)``.
        """
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Two convolutions with an optional residual connection.

    The residual is only wired up when the shapes allow it (``c1 == c2``) so
    input and output channel(s) are the same.

    Args:
        c1: Input channels.
        c2: Output channels.
        shortcut: The residual connection. Honored only if ``c1 == c2``.
        k: One kernel config per convolution, as ``(cv1, cv2)``. Each entry is whatever
            ``Conv`` takes as param.
        e: Hidden-channel ratio. The width between the two convs is ``int(c2 * e)``. At
            the default ``0.5`` the pair costs about as much as a single 3x3 conv;
            ``1.0`` removes the squeeze and doubles that.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        k: tuple[int | tuple[int, int], int | tuple[int, int]] = (3, 3),
        e: float = 0.5,
    ):
        super().__init__()

        # Bottleneck
        c_ = int(c2 * e)

        # Configure components
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the configured components.

        Args:
            x: ``(B, c1, H, W)`` feature map.

        Returns:
            ``(B, c2, H, W)`` -- both convs are stride 1, so the resolution is unchanged.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """A cross stage partial block (CSP). ``cv1`` projects the input and splits it in
    two. One half runs through ``n`` bottlenecks, keeping every intermediate
    result. The resulting ``n + 2`` branches are concatenated and fused back to ``c2``
    by a 1x1 convolution.

    Args:
        c1: Input channels.
        c2: Output channels.
        n: Number of chained bottlenecks, so ``n + 2`` branches reach the fusing convolution.
        shortcut: Passed to every inner ``Bottleneck``.
        e: Branch width ratio -- each branch carries ``int(c2 * e)`` channels.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)

        # Configure components
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # n times the Bottleneck
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the configured components.

        Save the intermediate results and concat them in the end.

        Args:
            x: ``(B, c1, H, W)`` feature map.

        Returns:
            ``(B, c2, H, W)``.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """A cross stage partial block (CSP) similar to ``C2F``, but here only the last
    bottleneck result is concatenated and fused back to ``c2`` by a 1x1 convolution.

    Args:
        c1: Input channels.
        c2: Output channels.
        n: Number of bottlenecks in the processed half.
        shortcut: Passed to every inner ``Bottleneck``.
        e: Width ratio of *both* halves -- each carries ``int(c2 * e)`` channels.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)

        # Configure components
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)

        # n times the Bottleneck
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the configured components.

        Args:
            x: ``(B, c1, H, W)`` feature map.

        Returns:
            ``(B, c2, H, W)``.
        """
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """A ``C3`` whose inner bottlenecks use 3x3 for both convolutions instead of 1x1 then
    3x3.

    The parameters are exactly ``C3``'s and mean the same thing. Only ``self.m`` differs.

    Args:
        c1: Input channels.
        c2: Output channels.
        n: Number of bottlenecks in the processed half.
        shortcut: Passed to every inner ``Bottleneck``.
        e: Width ratio of both halves -- each carries ``int(c2 * e)`` channels.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, e: float = 0.5):
        super().__init__(c1, c2, n, shortcut, e)
        c_ = int(c2 * e)

        # n times the Bottleneck, replacing the 1x1-then-3x3 stack ``C3`` just built
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, k=(3, 3), e=1.0) for _ in range(n)))


class C3k2(C2f):
    """A ``C2f`` whose inner block is swappable:
    ``attn`` gives a ``Bottleneck`` followed by a ``PSABlock``, ``c3k`` a nested ``C3k``
    of two bottlenecks, and neither a plain ``Bottleneck``. ``attn`` is checked first, so
    it wins over ``c3k``.

    Args:
        c1: Input channels.
        c2: Output channels.
        n: Number of inner blocks, so ``n + 2`` branches reach the fusing convolution.
        c3k: Use a nested ``C3k`` as the inner block. Ignored when ``attn`` is set.
        e: Branch width ratio -- each branch carries ``int(c2 * e)`` channels. Distinct
            from the inner ``Bottleneck``'s own ``e``, per the note above.
        attn: Use ``Bottleneck`` + ``PSABlock`` as the inner block.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, attn: bool = False):
        super().__init__(c1, c2, n, shortcut=True, e=e)

        def block():
            if attn:
                return nn.Sequential(
                    Bottleneck(self.c, self.c, True),
                    PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
                )
            if c3k:
                return C3k(self.c, self.c, 2, True)
            return Bottleneck(self.c, self.c, True)

        # n times the inner block
        self.m = nn.ModuleList(block() for _ in range(n))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast. Applies the same ``k x k`` max-pool ``n`` times in
    sequence and concatenates every stage, so the ``n + 1`` branches reaching the fusing
    convolution see progressively wider context.

    Args:
        c1: Input channels. The hidden width is ``c1 // 2``.
        c2: Output channels.
        k: Max-pool kernel size. Stride 1 and padding ``k // 2`` keep the resolution.
        n: How many times the pool is applied, giving ``n + 1`` concatenated branches.
        shortcut: Add the input to the output. Honored only if ``c1 == c2``.
    """

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        super().__init__()
        c_ = c1 // 2

        # Configure components
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the configured components.

        Args:
            x: ``(B, c1, H, W)`` feature map.

        Returns:
            ``(B, c2, H, W)``.
        """
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        y = self.cv2(torch.cat(y, 1))

        return y + x if self.add else y


class Attention(nn.Module):
    """Multi-head self-attention over all pixels, plus a depthwise convolution.

    Args:
        dim: Channels in and out. Must be divisible by ``num_heads``.
        num_heads: Attention heads. Each gets ``head_dim = dim // num_heads`` value
            channels.
        attn_ratio: How much narrower the queries and keys are than the values --
            ``key_dim = int(head_dim * attn_ratio)``.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()

        # Head geometry
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2  # q and k are narrower than v, so this is not 3 * dim

        # Configure components
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the configured components.

        Args:
            x: ``(B, dim, H, W)`` feature map.

        Returns:
            ``(B, dim, H, W)``.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q * self.scale).transpose(-2, -1) @ k
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)


class PSABlock(nn.Module):
    """A transformer block using Attention and Convolution.

    Args:
        c: Channels in and out. Both residuals require the width to be preserved.
        attn_ratio: Forwarded to ``Attention``.
        num_heads: Forwarded to ``Attention``. ``c`` must be divisible by it.
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4):
        super().__init__()

        # Configure components
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the configured components.

        Args:
            x: ``(B, c, H, W)`` feature map.

        Returns:
            ``(B, c, H, W)``.
        """
        x = x + self.attn(x)
        return x + self.ffn(x)


class C2PSA(nn.Module):
    """A cross stage partial block (CSP) with attention. ``cv1`` projects the input and
    splits it in two. One half runs through ``n`` stacked ``PSABlock``s, the other skips
    them, and both are concatenated and fused back to ``c`` by a 1x1 convolution.

    Args:
        c: Channels in and out.
        n: Number of stacked ``PSABlock``s the attended half passes through.
        e: Fraction of the width routed through attention -- ``int(c * e)`` channels are
            attended and the same number bypass the stack.
    """

    def __init__(self, c: int, n: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c * e)

        # Configure components
        self.cv1 = Conv(c, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c, 1)

        # n times the PSABlock
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the configured components.

        Only the second half is attended. The first reaches the fusing convolution
        unchanged.

        Args:
            x: ``(B, c, H, W)`` feature map.

        Returns:
            ``(B, c, H, W)``.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
