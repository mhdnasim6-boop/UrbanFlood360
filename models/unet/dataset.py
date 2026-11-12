"""
UrbanFlood360 MODIFICATION (not repository code).

Dataset loader for the validated UrbanFlood360 patch dataset
(data/processed/unet_dataset/patches/{split}/{images,labels}/*.npy).

This replaces the original repository's Sentinel1Dataset (which read 4
separate per-band GeoTIFFs with a fixed physical-range normalization)
because UrbanFlood360 patches are already-stacked 6-channel .npy arrays
with a different, train-fitted normalization strategy (see normalize.py).

Label semantics are preserved exactly as produced by create_flood_labels.py:
  0 = non-flood, 1 = flood, 255 = ignore.
The label is cast to float32 (not remapped) so the repository's existing
`y.ne(255)` masking pattern in train.py works unchanged.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class UrbanFlood360PatchDataset(Dataset):
    def __init__(self, patches_dir: str, split: str, normalizer=None):
        self.images_dir = Path(patches_dir) / split / "images"
        self.labels_dir = Path(patches_dir) / split / "labels"
        self.patch_ids = sorted(p.stem for p in self.images_dir.glob("*.npy"))
        self.normalizer = normalizer
        if not self.patch_ids:
            raise RuntimeError(f"No patches found in {self.images_dir}")

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        pid = self.patch_ids[idx]
        img = np.load(self.images_dir / f"{pid}.npy")  # (H,W,6) float32, may contain NaN
        lbl = np.load(self.labels_dir / f"{pid}.npy")  # (H,W) uint8, values {0,1,255}

        nan_mask = np.isnan(img)
        n_nan = int(nan_mask.sum())

        if self.normalizer is not None:
            img = self.normalizer.apply(img)

        # Fill NaN with 0.0 AFTER normalization (0.0 = the fitted per-channel mean
        # in z-score space, a neutral value). This must happen before the tensor
        # reaches any convolution: unlike the loss mask (which only protects the
        # loss computation), a NaN pixel left in the input would corrupt every
        # output pixel whose receptive field overlaps it, via convolution mixing.
        img = np.where(np.isnan(img), 0.0, img).astype("float32")

        image = torch.from_numpy(np.transpose(img, (2, 0, 1)))  # -> (6,H,W)
        label = torch.from_numpy(lbl.astype("float32"))          # (H,W), values 0/1/255
        label = torch.unsqueeze(label, 0)                        # (1,H,W), matches repo convention

        return image, label, n_nan, pid
