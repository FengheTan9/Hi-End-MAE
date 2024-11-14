from typing import *

import torch
from torch import nn

from downstream.models.vox2vec.default_params import *
from .blocks import ResBlock3d, StackMoreLayers


class FPN3d(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            base_channels: int, 
            num_scales: int, 
            deep: bool = False
    ) -> None:
        """Feature Pyramid Network (FPN) with 3D UNet architecture.

        Args:
            in_channels (int, optional):
                Number of input channels.
            out_channels (int, optional):
                Number of channels in the base of output feature pyramid.
            num_scales (int, optional):
                Number of pyramid levels.
            deep (bool):
                If True, add more layers at the bottom levels of UNet.
        """
        super().__init__()

        c = base_channels
        self.first_conv = nn.Conv3d(in_channels, c, kernel_size=3, padding=1)

        left_blocks, down_blocks, up_blocks, skip_blocks, right_blocks = [], [], [], [], []
        num_blocks = 2  # default
        for i in range(num_scales - 1):
            if deep:
                if i >= 2:
                    num_blocks = 4
                if i >= 4:
                    num_blocks = 8

            left_blocks.append(StackMoreLayers(ResBlock3d, [c] * (num_blocks + 1), kernel_size=3, padding=1))
            down_blocks.append(nn.Sequential(
                nn.MaxPool3d(kernel_size=2, ceil_mode=True),
                nn.Conv3d(c, c * 2, kernel_size=1)
            ))
            up_blocks.insert(0, nn.Sequential(
                nn.Conv3d(c * 2, c, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='nearest')
            ))
            skip_blocks.insert(0, nn.Conv3d(c, c, kernel_size=1))
            right_blocks.insert(0, StackMoreLayers(ResBlock3d, [c] * (num_blocks + 1), kernel_size=3, padding=1))

            c *= 2

        self.left_blocks = nn.ModuleList(left_blocks)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bottom_block = StackMoreLayers(ResBlock3d, [c] * (num_blocks + 1), kernel_size=3, padding=1)
        self.up_blocks = nn.ModuleList(up_blocks)
        self.skip_blocks = nn.ModuleList(skip_blocks)
        self.right_blocks = nn.ModuleList(right_blocks)
        self.base_channels = base_channels
        self.num_scales = num_scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_conv(x)

        feature_pyramid = []
        for left, down in zip(self.left_blocks, self.down_blocks):
            x = left(x)
            feature_pyramid.append(x)
            x = down(x)

        x = self.bottom_block(x)
        feature_pyramid.insert(0, x)

        for up, skip, right in zip(self.up_blocks, self.skip_blocks, self.right_blocks):
            x = up(x)
            fmap = feature_pyramid.pop()
            x = x[(..., *map(slice, fmap.shape[-3:]))]
            x += skip(fmap)  # skip connection
            x = right(x)
            feature_pyramid.insert(0, x)

        return feature_pyramid


class FPNLinearHead(nn.Module):
    def __init__(self, base_channels: int, num_scales: int, num_classes: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Conv3d(base_channels * 2 ** i, num_classes, kernel_size=1, bias=(i == 0))
            for i in range(num_scales)
        ])
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.num_scales = num_scales

    def forward(self, feature_pyramid: Sequence[torch.Tensor]) -> torch.Tensor:
        assert len(feature_pyramid) == self.num_scales

        feature_pyramid = [layer(x) for x, layer in zip(feature_pyramid, self.layers)]

        x = feature_pyramid[-1]
        for fmap in reversed(feature_pyramid[:-1]):
            x = self.up(x)
            x = x[(..., *map(slice, fmap.shape[-3:]))]
            x += fmap
        return x


class FPNNonLinearHead(nn.Module):
    def __init__(self, base_channels: int, num_scales: int, num_classes: int) -> None:
        super().__init__()

        c = base_channels
        up_blocks, skip_blocks, right_blocks = [], [], []
        for _ in range(num_scales - 1):
            up_blocks.insert(0, nn.Sequential(
                nn.Conv3d(c * 2, c, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='nearest')
            ))
            skip_blocks.insert(0, nn.Conv3d(c, c, kernel_size=1))
            right_blocks.insert(0, ResBlock3d(c, c, kernel_size=1))
            c *= 2

        self.bottom_block = ResBlock3d(c, c, kernel_size=1)
        self.up_blocks = nn.ModuleList(up_blocks)
        self.skip_blocks = nn.ModuleList(skip_blocks)
        self.right_blocks = nn.ModuleList(right_blocks)
        self.final_block = nn.Conv3d(base_channels, num_classes, kernel_size=1)
        self.num_scales = num_scales

    def forward(self, feature_pyramid: Sequence[torch.Tensor]) -> torch.Tensor:
        assert len(feature_pyramid) == self.num_scales

        x = feature_pyramid[-1]
        x = self.bottom_block(x)
        for up, skip, right, fmap in zip(self.up_blocks, self.skip_blocks, self.right_blocks,
                                         reversed(feature_pyramid[:-1])):
            x = up(x)
            x = x[(..., *map(slice, fmap.shape[-3:]))]
            x += skip(fmap)  # skip connection
            x = right(x)

        x = self.final_block(x)

        return x
