"""Seeded synthetic datasets for the procedure runner. Every dataset says what it is: the data is generated, not collected."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .spec import DataSpec

SYNTHETIC_NOTE = "SYNTHETIC data generated from data_seed; it stands for no real measurement."


@dataclass
class Dataset:
    fields: dict[str, torch.Tensor]
    description: str
    synthetic: bool = True

    def __len__(self) -> int:
        return len(next(iter(self.fields.values())))

    def batch(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.fields.items()}


def _sequence(n: int, spec: DataSpec, g: torch.Generator) -> Dataset:
    """Token sequences with padding id 0. Label 1 if token 1 occurs more often than token 2, else 0 (ties are broken so the rule is exact)."""
    t, v = spec.seq_len, spec.vocab
    tokens = torch.randint(1, v, (n, t), generator=g)
    lengths = torch.randint(max(2, t // 2), t + 1, (n,), generator=g)
    pad = torch.arange(t)[None, :] >= lengths[:, None]
    tokens = tokens.masked_fill(pad, 0)
    c1, c2 = (tokens == 1).sum(1), (tokens == 2).sum(1)
    for i in torch.nonzero(c1 == c2).flatten().tolist():  # exact rule: break ties by setting the first real token
        tokens[i, 0] = 1 if torch.rand(1, generator=g).item() < 0.5 else 2
    c1, c2 = (tokens == 1).sum(1), (tokens == 2).sum(1)
    y = (c1 > c2).to(torch.int64)
    return Dataset({"x": tokens, "y": y}, f"token sequences [N,{t}] (id 0 = padding, ids 1..{v - 1}); label = more 1s than 2s. {SYNTHETIC_NOTE}")


def _regression(n: int, spec: DataSpec, g: torch.Generator, w: torch.Tensor) -> Dataset:
    x = torch.randn(n, spec.features, generator=g)
    y = x @ w + 0.1 * torch.randn(n, spec.outputs, generator=g)
    mask = (torch.rand(n, spec.outputs, generator=g) > 0.25).float()  # 1 = this target is observed
    weights = torch.tensor([0.5, 1.0, 2.0])[torch.randint(0, 3, (n, spec.outputs), generator=g)]
    return Dataset({"x": x, "y": y, "mask": mask, "weights": weights},
                   f"regression x[N,{spec.features}] -> y[N,{spec.outputs}] with a validity mask (about 25% missing targets) and per-target weights. {SYNTHETIC_NOTE}")


def make_datasets(spec: DataSpec) -> tuple[Dataset, Dataset]:
    g = torch.Generator().manual_seed(spec.data_seed)
    if spec.kind == "synthetic_sequence":
        return _sequence(spec.n_train, spec, g), _sequence(spec.n_val, spec, g)
    if spec.kind == "synthetic_regression":
        w = torch.randn(spec.features, spec.outputs, generator=g)
        return _regression(spec.n_train, spec, g, w), _regression(spec.n_val, spec, g, w)
    if spec.kind == "image_folder":
        from worker.dataset import load_image_folder

        d = load_image_folder(spec.path, spec.image_size[0], spec.image_size[1], spec.data_seed, spec.val_fraction)
        note = f"image folder {spec.path} ({len(d.classes)} classes: {', '.join(d.classes)}), images resized to {spec.image_size[0]}x{spec.image_size[1]}, scaled to [-1, 1]; split seed {spec.data_seed}"
        return Dataset({"x": d.x_train, "y": d.y_train}, note, synthetic=False), Dataset({"x": d.x_val, "y": d.y_val}, note, synthetic=False)
    raise ValueError("dataset kind 'provided' needs datasets passed to the Trainer")
