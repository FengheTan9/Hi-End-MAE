from typing import *

import torch
from torch import nn

from models.vox2vec.nn.functional import eval_mode, sw_predict


@torch.no_grad()
def predict(
        image: torch.Tensor,
        patch_size: Tuple[int, int, int],
        backbone: nn.Module,
        head: nn.Module,
        device: str,
        roi: Optional[torch.Tensor] = None
) -> torch.Tensor:
    backbone.to(device)
    head.to(device)

    def predictor(x):
        return torch.sigmoid(head(backbone(x.to(device))).cpu())

    image = image.unsqueeze(0)

    with eval_mode(backbone), eval_mode(head):
        pred_probas = sw_predict(image, predictor, patch_size)

    pred_probas.squeeze_(0)

    if roi is not None:
        pred_probas *= roi

    return pred_probas
