"""
Ignore-aware losses for U-Net v2. Pixels labelled 255 (outside AOI / outside
the EMSR773 image footprint / "flood trace" / SAR-invalid) contribute nothing.

Default loss = 0.5 * weighted BCE + 0.5 * soft Dice. This is the combination
that gave the best validation Dice in the archived loss experiment
(archive/unet_experiments/reports/unet_loss_experiment_report.md).
"""
import torch
import torch.nn.functional as F

IGNORE_LABEL = 255


def masked_bce_dice(logits, target, pos_weight=1.0, bce_weight=0.5, eps=1.0):
    valid = target.ne(IGNORE_LABEL)
    if valid.sum() == 0:
        return logits.sum() * 0.0
    z = logits[valid]
    y = target[valid].float()

    bce = F.binary_cross_entropy_with_logits(
        z, y, pos_weight=torch.as_tensor(pos_weight, device=z.device, dtype=z.dtype)
    )
    p = torch.sigmoid(z)
    dice = 1 - (2 * (p * y).sum() + eps) / (p.sum() + y.sum() + eps)
    return bce_weight * bce + (1 - bce_weight) * dice
