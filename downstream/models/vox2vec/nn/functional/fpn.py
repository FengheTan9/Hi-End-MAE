from typing import *

import torch


def select_from_pyramid(
        feature_pyramid: Sequence[torch.Tensor],
        indices: torch.Tensor,
) -> torch.Tensor:
    """Select features from feature pyramid by their indices w.r.t. base feature map.

    Args:
        feature_pyramid (Sequence[torch.Tensor]): Sequence of tensors of shapes ``(c_i, h_i, w_i, d_i)``.
        indices (torch.Tensor): tensor of shape ``(n, 3)``

    Returns:
        torch.Tensor: tensor of shape ``(n, \sum_i c_i)``
    """
    return torch.cat([x.moveaxis(0, -1)[(indices // 2 ** i).unbind(1)] for i, x in enumerate(feature_pyramid)], dim=1)


def sum_pyramid_channels(base_channels: int, num_scales: int):
    return sum(base_channels * 2 ** i for i in range(num_scales))
