"""Geometric transforms that keep every annotation consistent with the image (A56). The geometry is torchvision.transforms.v2.functional on tv_tensors
(Image, Mask, BoundingBoxes with clamping disabled, KeyPoints); this module adds what torchvision leaves to the caller and makes it declared and visible:

- the box policy (clip to the canvas, remove an instance whose visible fraction or size falls below a threshold) applied after EVERY step, with the removed instances logged;
- keypoint flip-pair swapping when pairs are declared, and visibility 0 for keypoints that leave the canvas;
- instance rows (box, label, mask, keypoints) are always removed together;
- boxes stay in the DECLARED format/coordinates: they are converted to xyxy pixels for the step and back."""
from __future__ import annotations

from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from .contract import (ContractError, tight_box, VisionSample, VisionSpec, from_xyxy_pixel, keypoints_from_pixel, keypoints_to_pixel, to_xyxy_pixel)


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["resize", "crop", "hflip", "vflip", "rot90", "pad", "affine", "random_hflip"]
    size: list[int] | None = Field(None, description="resize: [height, width]")
    top: int = 0
    left: int = 0
    height: int | None = None
    width: int | None = None
    k: int = Field(1, description="rot90: number of counter-clockwise quarter turns (1-3)")
    pad: list[int] = Field(default_factory=lambda: [0, 0, 0, 0], description="pad: [left, top, right, bottom]")
    fill: int = 0
    angle: float = 0.0
    translate: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    scale: float = 1.0
    shear: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    p: float = 0.5
    seed: int = 0


class BoxPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clip: bool = Field(True, title="clip boxes to the canvas after every step")
    min_visible_fraction: float = Field(0.0, ge=0, le=1, title="remove an instance whose clipped area / unclipped area (entering the step) is below this")
    min_size: float = Field(1.0, ge=0, title="remove an instance whose clipped width or height is below this many pixels")
    refit_to_mask: bool = Field(False, title="after clipping, replace each box by the tight bounds of its remaining mask (clipping a box can leave it larger than the visible object)")


def out_size(step: Step, size: tuple[int, int]) -> tuple[int, int]:
    """Canvas size after a step (static: used by validation as well)."""
    h, w = size
    if step.op == "resize":
        if not step.size or len(step.size) != 2:
            raise ContractError("E_VISION_TRANSFORM_PARAM", "resize needs size=[height, width]")
        return step.size[0], step.size[1]
    if step.op == "crop":
        ch, cw = step.height or 0, step.width or 0
        if ch < 1 or cw < 1:
            raise ContractError("E_VISION_TRANSFORM_PARAM", "crop needs height and width >= 1")
        if step.top < 0 or step.left < 0 or step.top + ch > h or step.left + cw > w:
            raise ContractError("E_VISION_TRANSFORM_CROP", f"crop window (top {step.top}, left {step.left}, {ch}x{cw}) is not inside the {h}x{w} canvas")
        return ch, cw
    if step.op == "rot90":
        if step.k not in (1, 2, 3):
            raise ContractError("E_VISION_TRANSFORM_PARAM", "rot90 k must be 1, 2 or 3")
        return (w, h) if step.k % 2 else (h, w)
    if step.op == "pad":
        if len(step.pad) != 4 or min(step.pad) < 0:
            raise ContractError("E_VISION_TRANSFORM_PARAM", "pad needs [left, top, right, bottom] >= 0")
        return h + step.pad[1] + step.pad[3], w + step.pad[0] + step.pad[2]
    return h, w


def _swap(t: torch.Tensor, pairs: list[list[int]]) -> torch.Tensor:
    t = t.clone()
    for a, b in pairs:
        t[:, [a, b]] = t[:, [b, a]]
    return t


def _geometry(step: Step, image, boxes, masks, kps, size):
    """One step on tv_tensors; returns the transformed tensors."""
    h, w = size
    if step.op == "resize":
        nh, nw = out_size(step, size)
        fn = lambda x: F.resize(x, [nh, nw], antialias=isinstance(x, tv_tensors.Image))
    elif step.op == "crop":
        fn = lambda x: F.crop(x, step.top, step.left, step.height, step.width)
    elif step.op == "hflip":
        fn = F.horizontal_flip
    elif step.op == "vflip":
        fn = F.vertical_flip
    elif step.op == "rot90":
        fn = lambda x: F.rotate(x, 90.0 * step.k, expand=True, interpolation=F.InterpolationMode.NEAREST)
    elif step.op == "pad":
        fn = lambda x: F.pad(x, list(step.pad), fill=step.fill if isinstance(x, tv_tensors.Image) else 0)
    elif step.op == "affine":
        fn = lambda x: F.affine(x, angle=step.angle, translate=list(step.translate), scale=step.scale, shear=list(step.shear), interpolation=F.InterpolationMode.NEAREST)
    else:
        raise ContractError("E_VISION_TRANSFORM_PARAM", f"unknown step {step.op}")
    return fn(image), (fn(boxes) if boxes is not None else None), (fn(masks) if masks is not None else None), (fn(kps) if kps is not None else None)


def kp_src_flip(k: torch.Tensor, op: str, size: tuple[int, int]) -> torch.Tensor:
    k = k.clone()
    if op == "hflip":
        k[..., 0] = size[1] - k[..., 0]
    else:
        k[..., 1] = size[0] - k[..., 1]
    return k


def apply_step(s: VisionSample, spec: VisionSpec, step: Step, policy: BoxPolicy, index: int = 0) -> tuple[VisionSample, dict[str, Any]]:
    """Apply one step to one sample. Returns the new sample and a log entry with the ACTUAL parameters and the instances removed."""
    h, w = s.size
    n = len(s.boxes)
    op, applied = step.op, True
    if op == "random_hflip":
        g = torch.Generator().manual_seed(step.seed * 1_000_003 + index)
        applied = bool(torch.rand(1, generator=g).item() < step.p)
        op = "hflip"
    log: dict[str, Any] = {"op": step.op, "applied": applied, "params": step.model_dump(exclude_defaults=True), "removed": [], "canvasBefore": [h, w]}
    st = step if step.op != "random_hflip" else Step(op="hflip")
    if not applied:
        log["canvasAfter"] = [h, w]
        return s.clone(), log
    xyxy = to_xyxy_pixel(s.boxes, spec.box_format, spec.coords, (h, w))
    tv_boxes = tv_tensors.BoundingBoxes(xyxy, format="XYXY", canvas_size=(h, w), clamping_mode=None)
    kp_px = keypoints_to_pixel(s.keypoints, spec.coords, (h, w))
    tv_kps = tv_tensors.KeyPoints(kp_px, canvas_size=(h, w)) if kp_px is not None else None
    tv_masks = tv_tensors.Mask(s.masks) if s.masks is not None else None
    img, bx, mk, kp = _geometry(st, tv_tensors.Image(s.image), tv_boxes, tv_masks, tv_kps, (h, w))
    nh, nw = int(img.shape[-2]), int(img.shape[-1])
    bx = torch.as_tensor(bx).clone()
    vis = None if s.visibility is None else s.visibility.clone()
    kpt = None if kp is None else torch.as_tensor(kp).clone()
    if kpt is not None and op in ("hflip", "vflip"):
        # torchvision's KeyPoints flip uses (size - 1 - x), i.e. integer pixel centres, while its resize/rotate/pad/crop/affine (and boxes) use continuous coordinates
        # (pixel i spans [i, i+1)). One convention is declared here (continuous), so flips are computed explicitly: x' = W - x (hflip), y' = H - y (vflip).
        kpt = kp_src_flip(kp_px, op, (nh, nw))
    if kpt is not None and op == "hflip" and spec.flip_pairs:
        kpt, vis = _swap(kpt, spec.flip_pairs), _swap(vis, spec.flip_pairs)
        log["keypointPairsSwapped"] = spec.flip_pairs
    if kpt is not None:
        outside = (kpt[..., 0] < 0) | (kpt[..., 0] >= nw) | (kpt[..., 1] < 0) | (kpt[..., 1] >= nh)
        newly = int(((vis > 0) & outside).sum())
        vis = torch.where(outside, torch.zeros_like(vis), vis)
        log["keypointsMarkedInvisible"] = newly
    # ---- box policy: relative to the unclipped box that entered the policy
    keep = torch.ones(n, dtype=torch.bool)
    area0 = ((bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])).clamp(min=0) if n else bx[:, 0]
    clipped = bx.clone()
    if n:
        clipped[:, [0, 2]] = clipped[:, [0, 2]].clamp(0, nw)
        clipped[:, [1, 3]] = clipped[:, [1, 3]].clamp(0, nh)
        cw, ch = clipped[:, 2] - clipped[:, 0], clipped[:, 3] - clipped[:, 1]
        area1 = cw.clamp(min=0) * ch.clamp(min=0)
        frac = torch.where(area0 > 0, area1 / area0.clamp(min=1e-9), torch.zeros_like(area1))
        why = {}
        for i in range(n):
            if area1[i] <= 0:
                why[i] = "no visible area left in the canvas"
            elif frac[i] < policy.min_visible_fraction:
                why[i] = f"visible fraction {float(frac[i]):.3f} < {policy.min_visible_fraction}"
            elif policy.min_size > 0 and (cw[i] < policy.min_size or ch[i] < policy.min_size):
                why[i] = f"clipped size {float(cw[i]):.1f}x{float(ch[i]):.1f} < {policy.min_size}px"
        for i, msg in why.items():
            keep[i] = False
            log["removed"].append({"instance": i, "label": int(s.labels[i]), "reason": msg})
        if policy.clip:
            bx = clipped
    mk_t = None if mk is None else torch.as_tensor(mk).to(torch.bool)
    if n and mk_t is not None:
        for i in range(n):
            if not keep[i]:
                continue
            tb = tight_box(mk_t[i])
            if tb is None:
                keep[i] = False
                log["removed"].append({"instance": i, "label": int(s.labels[i]), "reason": "its mask has no pixels left in the canvas"})
            elif policy.refit_to_mask:
                bx[i] = torch.tensor(tb)
                log["refit"] = True
    out = VisionSample(torch.as_tensor(img).clone(), from_xyxy_pixel(bx, spec.box_format, spec.coords, (nh, nw))[keep], s.labels[keep],
                       None if mk_t is None else mk_t[keep], None if kpt is None else keypoints_from_pixel(kpt, spec.coords, (nh, nw))[keep], None if vis is None else vis[keep], dict(s.meta))
    log["canvasAfter"] = [nh, nw]
    log["instancesBefore"], log["instancesAfter"] = n, int(keep.sum())
    return out, log


def apply_pipeline(s: VisionSample, spec: VisionSpec, steps: list[Step], policy: BoxPolicy, index: int = 0) -> tuple[VisionSample, list[dict[str, Any]]]:
    logs = []
    for st in steps:
        s, lg = apply_step(s, spec, st, policy, index)
        logs.append(lg)
    return s, logs


def static_size(steps: list[Step], size: tuple[int, int]) -> tuple[int, int]:
    for st in steps:
        size = out_size(st, size)
    return size
