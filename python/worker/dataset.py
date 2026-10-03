"""ImageFolder-style dataset: root/<class>/<image>. Deterministic seeded train/val split."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class Split:
    classes: list[str]
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    files_sha256: str  # identity of the (sorted) file list and its contents
    train_files: list[str]  # paths relative to root, in x_train order
    val_files: list[str]  # paths relative to root, in x_val order (val sample id = index into this list)
    split_seed: int
    val_fraction: float


def preprocess_image(im: Image.Image, height: int, width: int) -> torch.Tensor:
    """The one transform used for training, evaluation and inference: RGB, bilinear resize, [-1, 1]. Returns [3,H,W]."""
    im = im.convert("RGB").resize((width, height), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
    return (x - 0.5) / 0.5


def load_image_folder(root: str | Path, height: int, width: int, seed: int, val_fraction: float) -> Split:
    root = Path(root)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not classes:
        raise ValueError(f"no class subdirectories under {root}")
    files, labels = [], []
    for ci, c in enumerate(classes):
        for f in sorted((root / c).iterdir()):
            if f.suffix.lower() in IMAGE_EXT:
                files.append(f)
                labels.append(ci)
    if len(files) < 2:
        raise ValueError(f"need at least 2 images under {root}")
    digest = hashlib.sha256()
    imgs = []
    for f in files:
        digest.update(str(f.relative_to(root)).encode())
        digest.update(f.read_bytes())
        with Image.open(f) as im:
            imgs.append(preprocess_image(im, height, width))
    x = torch.stack(imgs)
    y = torch.tensor(labels, dtype=torch.int64)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(files), generator=g)
    n_val = min(max(1, round(len(files) * val_fraction)), len(files) - 1)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    rel = [str(f.relative_to(root)) for f in files]
    return Split(classes, x[train_idx], y[train_idx], x[val_idx], y[val_idx], digest.hexdigest(),
                 [rel[i] for i in train_idx.tolist()], [rel[i] for i in val_idx.tolist()], seed, val_fraction)
