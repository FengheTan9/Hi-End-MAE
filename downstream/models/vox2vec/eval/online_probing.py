from typing import *

import torch
from torch import nn

import pytorch_lightning as pl

from vox2vec.nn.functional import (
    compute_binary_segmentation_loss, compute_dice_score, eval_mode
)
from .predict import predict


class OnlineProbing(pl.Callback):
    def __init__(
            self,
            *heads: nn.Module,
            patch_size: Tuple[int, int, int],
            lr: float = 3e-4
    ):
        self.heads = nn.ModuleList(heads)
        self.optimizer = torch.optim.Adam(self.heads.parameters(), lr=lr)

        self.patch_size = patch_size

    def on_fit_start(self, trainer, pl_module):
        self.heads.to(pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.optimizer.zero_grad()

        images, rois, gt_masks = batch['online_probing']

        with torch.no_grad(), eval_mode(pl_module.backbone):
            backbone_outputs = pl_module.backbone(images)

        for i, head in enumerate(self.heads):
            pred_logits = head(backbone_outputs)
            loss, logs = compute_binary_segmentation_loss(pred_logits, gt_masks, rois, logs_prefix=f'train/head_{i}_')
            self.log_dict(logs)
            loss.backward()

        self.optimizer.step()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        image, roi, gt_mask = batch
        for i, head in enumerate(self.heads):
            pred_probas = predict(image, self.patch_size, pl_module.backbone, head, pl_module.device, roi)
            dice_scores = compute_dice_score(pred_probas, gt_mask, reduce=lambda x: x)
            for j, dice_score in enumerate(dice_scores):
                pl_module.log(f'val/head_{i}_dice_score_for_cls_{j}', dice_score, on_epoch=True)
            pl_module.log(f'val/head_{i}_avg_dice_score', dice_scores.mean(), on_epoch=True)
