from __future__ import annotations

import torch
import torch.nn.functional as F


IGNORE_INDEX = 255


def valid_label_mask(label: torch.Tensor) -> torch.Tensor:
    return label != IGNORE_INDEX


def masked_dice_loss(logits: torch.Tensor, label: torch.Tensor, num_classes: int) -> torch.Tensor:
    valid = valid_label_mask(label)
    if not valid.any():
        return logits.sum() * 0.0
    prob = torch.softmax(logits.float(), dim=1)
    safe_label = torch.where(valid, label, torch.zeros_like(label))
    target = F.one_hot(safe_label.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid_float = valid[:, None].float()
    prob = prob * valid_float
    target = target * valid_float
    intersection = (prob * target).sum(dim=(0, 2, 3))
    denominator = (prob * prob).sum(dim=(0, 2, 3)) + (target * target).sum(dim=(0, 2, 3))
    return 1.0 - ((2.0 * intersection + 1e-5) / (denominator + 1e-5)).mean()


def segmentation_loss(logits: torch.Tensor, label: torch.Tensor, num_classes: int) -> torch.Tensor:
    ce = F.cross_entropy(logits, label.long(), ignore_index=IGNORE_INDEX)
    return 0.5 * ce + 0.5 * masked_dice_loss(logits, label, num_classes)


def foreground_tversky_loss(
    logits: torch.Tensor,
    label: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    valid = valid_label_mask(label)
    if not valid.any():
        return logits.sum() * 0.0
    prob = torch.softmax(logits.float(), dim=1)[:, 1]
    target = (label == 1).float()
    valid_float = valid.float()
    prob = prob * valid_float
    target = target * valid_float
    tp = (prob * target).sum()
    fp = (prob * (1.0 - target) * valid_float).sum()
    fn = ((1.0 - prob) * target).sum()
    score = (tp + 1.0) / (tp + float(alpha) * fp + float(beta) * fn + 1.0)
    return 1.0 - score


def binary_dice_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    prob = torch.sigmoid(logits.float())
    target = target.float()
    if valid is not None:
        valid = valid.float()
        prob = prob * valid
        target = target * valid
    dims = tuple(range(1, prob.ndim))
    intersection = (prob * target).sum(dim=dims)
    denominator = prob.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def lgd_auxiliary_loss(aux_logits: list[torch.Tensor], label: torch.Tensor) -> torch.Tensor:
    if not aux_logits:
        return torch.zeros((), device=label.device)
    valid = valid_label_mask(label)[:, None].float()
    target_full = (label == 1)[:, None].float()
    losses = []
    for logits in aux_logits:
        target = F.interpolate(target_full, size=logits.shape[-2:], mode="nearest")
        target_valid = F.interpolate(valid, size=logits.shape[-2:], mode="nearest")
        bce_map = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
        bce = (bce_map * target_valid).sum() / target_valid.sum().clamp_min(1.0)
        dice = binary_dice_loss_from_logits(logits, target, target_valid)
        losses.append(bce + dice)
    return torch.stack(losses).mean()
def dilate_binary_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(0, int(radius))
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=radius).clamp(0.0, 1.0)


def lgd_localization_loss(
    locator_logits: list[torch.Tensor],
    label: torch.Tensor,
    dilate_radius: int,
) -> torch.Tensor:
    if not locator_logits:
        return torch.zeros((), device=label.device)

    valid_full = valid_label_mask(label)[:, None].float()
    target_full = dilate_binary_mask((label == 1)[:, None].float(), dilate_radius)
    losses = []
    for logits in locator_logits:
        target = F.interpolate(target_full, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        target_valid = F.interpolate(valid_full, size=logits.shape[-2:], mode="nearest")
        bce_map = F.binary_cross_entropy_with_logits(logits[:, :1].float(), target, reduction="none")
        bce = (bce_map * target_valid).sum() / target_valid.sum().clamp_min(1.0)
        dice = binary_dice_loss_from_logits(logits[:, :1], target, target_valid)
        losses.append(bce + dice)
    return torch.stack(losses).mean()
