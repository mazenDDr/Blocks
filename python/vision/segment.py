"""A tiny fully convolutional segmentation network, trained briefly on the SYNTHETIC shapes fixture, and its evaluation.

Segmentation metrics (per-class IoU, Dice, pixel accuracy) are accumulated over the whole validation set from one confusion matrix; they are checked against
torchmetrics in the tests. Detection metrics (mAP) come from torchmetrics' MeanAveragePrecision (pycocotools backend) on boxes read off the predicted masks
(CONNECTIVITY) with the mean class probability over the component as its score: the model is a segmenter, so these are derived detections and are labelled so."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from scipy import ndimage

from .contract import VisionSample, VisionSpec, to_xyxy_pixel

CONNECTIVITY = "4-connected components (scipy.ndimage.label default structure)"
PALETTE = [(0, 0, 0), (230, 80, 70), (70, 200, 110), (80, 120, 240), (240, 200, 60)]


class TinyFCN(nn.Module):
    """enc1 (conv-relu x2) -> pool -> enc2 (conv-relu x2) -> bilinear up -> concat enc1 -> conv -> 1x1 classifier (a 1-level U-shape)."""

    def __init__(self, n_classes: int, width: int = 12):
        super().__init__()
        w = width
        self.e1 = nn.Sequential(nn.Conv2d(3, w, 3, padding=1), nn.ReLU(), nn.Conv2d(w, w, 3, padding=1), nn.ReLU())
        self.e2 = nn.Sequential(nn.Conv2d(w, 2 * w, 3, padding=1), nn.ReLU(), nn.Conv2d(2 * w, 2 * w, 3, padding=1), nn.ReLU())
        self.d = nn.Sequential(nn.Conv2d(3 * w, w, 3, padding=1), nn.ReLU())
        self.head = nn.Conv2d(w, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.e1(x)
        b = self.e2(Fn.max_pool2d(a, 2))
        b = Fn.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.d(torch.cat([a, b], 1)))


def prep(images: torch.Tensor) -> torch.Tensor:
    return (images.float() / 255.0 - 0.5) / 0.25


def class_map(s: VisionSample) -> torch.Tensor:
    """(H, W) int64: 0 = background, c + 1 = an instance of class c (instance masks do not overlap in the fixture; the later instance wins)."""
    h, w = s.size
    cm = torch.zeros(h, w, dtype=torch.int64)
    if s.masks is not None:
        for i in range(len(s.labels)):
            cm[s.masks[i]] = int(s.labels[i]) + 1
    return cm


def confusion(pred: torch.Tensor, target: torch.Tensor, n: int) -> torch.Tensor:
    idx = target.reshape(-1) * n + pred.reshape(-1)
    return torch.bincount(idx, minlength=n * n).reshape(n, n)  # rows = target, columns = prediction


def seg_metrics(conf: torch.Tensor) -> dict[str, Any]:
    tp = conf.diag().double()
    fp = conf.sum(0).double() - tp
    fn = conf.sum(1).double() - tp
    union = tp + fp + fn
    present = (conf.sum(1) > 0) | (conf.sum(0) > 0)
    iou = torch.where(union > 0, tp / union.clamp(min=1), torch.full_like(tp, float("nan")))
    dice = torch.where(union > 0, 2 * tp / (2 * tp + fp + fn).clamp(min=1), torch.full_like(tp, float("nan")))
    return {"iou": iou.tolist(), "dice": dice.tolist(), "present": present.tolist(), "meanIoU": float(iou[present].mean()), "meanDice": float(dice[present].mean()),
            "pixelAccuracy": float(tp.sum() / conf.sum().clamp(min=1))}


def components(prob: torch.Tensor, pred: torch.Tensor, min_area: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derived detections: connected components of every predicted foreground class. prob (C,H,W) softmax, pred (H,W) argmax.
    Returns xyxy boxes (pixel edges, exclusive max), labels (class id without background), scores (mean class probability over the component)."""
    boxes, labels, scores = [], [], []
    for c in range(1, prob.shape[0]):
        lab, n = ndimage.label((pred == c).numpy())
        for k in range(1, n + 1):
            m = lab == k
            if m.sum() < min_area:
                continue
            ys, xs = np.nonzero(m)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            labels.append(c - 1)
            scores.append(float(prob[c].numpy()[m].mean()))
    return (torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), torch.tensor(labels, dtype=torch.int64), torch.tensor(scores, dtype=torch.float32))


def map_metrics(preds: list[dict], gts: list[dict]) -> dict[str, Any]:
    from torchmetrics.detection import MeanAveragePrecision

    m = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True, backend="pycocotools")
    m.update(preds, gts)
    r = m.compute()
    f = lambda k: None if k not in r or float(r[k]) < 0 else float(r[k])
    pc = r.get("map_per_class")
    classes = r.get("classes")
    return {"map": f("map"), "map50": f("map_50"), "map75": f("map_75"), "mapSmall": f("map_small"), "mapMedium": f("map_medium"), "mar100": f("mar_100"),
            "perClassAP": ({int(c): float(v) for c, v in zip(classes.tolist(), pc.tolist())} if pc is not None and pc.ndim else {})}


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    perm = np.random.default_rng(seed).permutation(n).tolist()
    nv = max(1, int(round(n * val_fraction)))
    return sorted(perm[nv:]), sorted(perm[:nv])


def stack(samples: list[VisionSample]) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.stack([s.image for s in samples]), torch.stack([class_map(s) for s in samples])


def train_and_eval(samples: list[VisionSample], spec: VisionSpec, *, epochs: int, lr: float, width: int, batch_size: int, seed: int, val_fraction: float,
                   n_inspect: int) -> dict[str, Any]:
    h, w = samples[0].size
    if any(s.size != (h, w) for s in samples):
        raise ValueError(f"all images must share one size to batch; got {sorted({s.size for s in samples})}")
    tr, va = split_indices(len(samples), val_fraction, seed)
    torch.manual_seed(seed)
    n_cls = len(spec.classes) + 1
    model = TinyFCN(n_cls, width)
    n_params = sum(p.numel() for p in model.parameters())
    xtr, ytr = stack([samples[i] for i in tr])
    xva, yva = stack([samples[i] for i in va])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    curve = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr), generator=g)
        tot = 0.0
        for k in range(0, len(tr), batch_size):
            b = perm[k:k + batch_size]
            loss = Fn.cross_entropy(model(prep(xtr[b])), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        model.eval()
        with torch.no_grad():
            vl = float(Fn.cross_entropy(model(prep(xva)), yva))
            vi = seg_metrics(confusion(model(prep(xva)).argmax(1), yva, n_cls))["meanIoU"]
        curve.append({"epoch": ep + 1, "trainLoss": tot / len(tr), "valLoss": vl, "valMeanIoU": vi})
    model.eval()
    with torch.no_grad():
        logits = model(prep(xva))
        prob = logits.softmax(1)
        pred = logits.argmax(1)
    seg = seg_metrics(confusion(pred, yva, n_cls))
    preds, gts, per_sample = [], [], []
    for j, i in enumerate(va):
        s = samples[i]
        pb, pl, ps = components(prob[j], pred[j])
        preds.append({"boxes": pb, "labels": pl, "scores": ps})
        gb = to_xyxy_pixel(s.boxes, spec.box_format, spec.coords, s.size)
        gts.append({"boxes": gb, "labels": s.labels})
        per_sample.append((pb, pl, ps, gb))
    det = map_metrics(preds, gts)
    recs = []
    for j, i in enumerate(va[:n_inspect]):
        s = samples[i]
        pb, pl, ps, gb = per_sample[j]
        c = confusion(pred[j], yva[j], n_cls)
        sm = seg_metrics(c)
        recs.append({"index": i, "id": s.meta.get("id"), "image": s.image.permute(1, 2, 0).numpy(), "gtMask": yva[j].numpy().astype(np.uint8), "predMask": pred[j].numpy().astype(np.uint8),
                     "gtBoxes": gb.tolist(), "gtLabels": s.labels.tolist(), "predBoxes": pb.tolist(), "predLabels": pl.tolist(), "predScores": ps.tolist(),
                     "gtKeypoints": None if s.keypoints is None else [[[float(x), float(y), int(v)] for (x, y), v in zip(k, vv)] for k, vv in zip(s.keypoints, s.visibility)],
                     "iou": sm["iou"], "pixelAccuracy": sm["pixelAccuracy"], "uncertainPixels": int(((prob[j].max(0).values < 0.6)).sum())})
    return {"nParams": n_params, "trainIdx": tr, "valIdx": va, "curve": curve, "seg": seg, "det": det, "records": recs, "seconds": time.time() - t0,
            "model": model, "nTrain": len(tr), "nVal": len(va)}
