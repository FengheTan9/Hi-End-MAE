from typing import *
import torch
import torch.nn.functional as F


def compute_dice_loss(
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        spatial_dims: Union[int, Sequence[int]] = (-3, -2, -1),
        reduce: Callable = torch.mean
) -> torch.Tensor:
    intersection = torch.sum(prediction * ground_truth, dim=spatial_dims)
    volumes_sum = torch.sum(prediction ** 2 + ground_truth ** 2, dim=spatial_dims)
    dice = 2 * intersection / (volumes_sum + 1)
    loss = 1 - dice
    return reduce(loss)


def compute_dice_score(
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        spatial_dims: Union[int, Sequence[int]] = (-3, -2, -1),
        reduce: Callable = torch.mean
) -> torch.Tensor:
    intersection = torch.sum(prediction * ground_truth, dim=spatial_dims)
    volumes_sum = torch.sum(prediction ** 2 + ground_truth ** 2, dim=spatial_dims)
    eps = 1e-5
    dice = (2 * intersection + eps) / (volumes_sum + eps)
    return reduce(dice)


def compute_binary_segmentation_loss(
        pred_logits: torch.Tensor,
        gt_masks: torch.Tensor,
        rois: Optional[torch.Tensor] = None,
        logs_prefix: str = '',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Average BCE and Dice losses per class.

    Args:
        pred_logits (torch.Tensor): tensor of shape ``(n, c, h, w, d)``.
        gt_masks (torch.Tensor): tensor of shape ``(n, c, h, w, d)``.
        rois (torch.Tensor): tensor of shape ``(n, h, w, d)``.

    Returns:
        Tuple[torch.Tensor, Dict[str, float]]: total loss and logs.
    """
    bce = F.binary_cross_entropy_with_logits(pred_logits, gt_masks, reduction='none')
    pred_probas = pred_logits.sigmoid()

    if rois is not None:
        rois = torch.broadcast_to(rois.unsqueeze(1), pred_logits.shape)
        bce = bce[rois]
        pred_probas = torch.where(rois, pred_probas, torch.tensor(0.0).to(pred_probas))
        gt_masks = torch.where(rois, gt_masks, torch.tensor(0.0).to(gt_masks))

    bce = bce.mean()
    dice_losses = compute_dice_loss(pred_probas, gt_masks, reduce=lambda x: x.mean(0))
    loss = bce + dice_losses.mean()
    logs = {
        f'{logs_prefix}bce': bce.item(),
        **{f'{logs_prefix}dice_loss_for_cls_{i}': l.item() for i, l in enumerate(dice_losses)},
        f'{logs_prefix}loss': loss.item()
    }
    return loss, logs


def compute_multiclass_segmentation_loss(
        pred_logits: torch.Tensor,
        gt_masks: torch.Tensor,
        rois: Optional[torch.Tensor] = None,
        logs_prefix: str = '',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Multiclass cross entropy and Dice losses.

    Args:
        pred_logits (torch.Tensor): tensor of size ``(n, c + 1, h, w, d)``. Raw logits.
        gt_masks (torch.Tensor): tensor of size ``(n, c + 1, h, w, d)``. One-hot mask.
        rois (torch.Tensor): tensor of size ``(n, h, w, d)``. Rois.

    Returns:
        Tuple[torch.Tensor, Dict[str, float]]: total loss and logs.
    """
    ce = F.cross_entropy(pred_logits, gt_masks, reduce='none')
    pred_probas = pred_logits.softmax(1)[:, 1:]  # (n, c, h, w, d)
    gt_masks = gt_masks[:, 1:]  # (n, c, h, w, d)

    if rois is not None:
        ce = ce[rois]
        pred_probas = torch.where(rois.unsqueeze(1), pred_logits.softmax(1)[:, 1:], torch.tensor(0.0).to(pred_logits))

    ce = ce.mean()
    dice_losses = compute_dice_loss(pred_probas, gt_masks, reduce=lambda x: x.mean(dim=0))
    loss = ce + dice_losses.mean()
    logs = {
        f'{logs_prefix}cross_entropy': ce.item(),
        **{f'{logs_prefix}dice_loss_for_cls_{i}': l.item() for i, l in enumerate(dice_losses)},
        f'{logs_prefix}loss': loss.item()
    }
    return loss, logs
