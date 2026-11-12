"""
UrbanFlood360 MODIFICATION (not repository code).

Ignore-aware (255) confusion-matrix accumulation and derived segmentation
metrics (Dice, IoU, precision, recall, specificity). Accumulates raw
TP/TN/FP/FN counts across an entire epoch (not a mean of per-batch ratios),
which is the statistically correct approach given several patches/batches
in this small dataset can contain zero positive pixels.
"""
import torch

IGNORE_LABEL = 255
EPS = 1e-7


class ConfusionAccumulator:
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def update(self, pred_prob: torch.Tensor, true: torch.Tensor, threshold: float = 0.5):
        valid_mask = true.ne(IGNORE_LABEL)
        pred_valid = pred_prob.masked_select(valid_mask)
        true_valid = true.masked_select(valid_mask)
        pred_bin = (pred_valid > threshold).float()

        self.tp += int(((pred_bin == 1) & (true_valid == 1)).sum().item())
        self.tn += int(((pred_bin == 0) & (true_valid == 0)).sum().item())
        self.fp += int(((pred_bin == 1) & (true_valid == 0)).sum().item())
        self.fn += int(((pred_bin == 0) & (true_valid == 1)).sum().item())

    def compute(self):
        tp, tn, fp, fn = self.tp, self.tn, self.fp, self.fn
        iou = tp / (tp + fp + fn + EPS)
        dice = 2 * tp / (2 * tp + fp + fn + EPS)
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        specificity = tn / (tn + fp + EPS)
        return dict(tp=tp, tn=tn, fp=fp, fn=fn, iou=iou, dice=dice,
                    precision=precision, recall=recall, specificity=specificity)
