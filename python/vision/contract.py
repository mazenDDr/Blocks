"""Detection / segmentation data contract: image + boxes (declared format and coordinate system) + instance masks + keypoints (with visibility) + metadata.

Boxes are stored in the DECLARED format (`xyxy`, `xywh` or `cxcywh`) and coordinates (`pixel` or `normalized` to the image width/height). Pixel coordinates are continuous:
pixel i spans [i, i+1), so a box covering columns 3..7 is x1=3, x2=8 (x2 exclusive) and a keypoint at the centre of pixel 5 is x=5.5. Keypoints use the same coordinate system
as the boxes. Visibility follows the COCO convention (0 not labelled / outside, 1 labelled but occluded, 2 visible)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

BOX_FORMATS = ("xyxy", "xywh", "cxcywh")
COORDS = ("pixel", "normalized")


class ContractError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


@dataclass
class VisionSpec:
    classes: list[str]
    keypoint_names: list[str] = field(default_factory=list)
    flip_pairs: list[list[int]] = field(default_factory=list)  # keypoint index pairs exchanged by a horizontal flip; [] = not declared
    box_format: str = "xyxy"
    coords: str = "pixel"
    color_space: str = "RGB"
    pixel_scale: str = "uint8 0-255"

    def to_json(self) -> dict[str, Any]:
        return {"classes": self.classes, "keypointNames": self.keypoint_names, "flipPairs": self.flip_pairs, "boxFormat": self.box_format, "coords": self.coords,
                "colorSpace": self.color_space, "channelOrder": "CHW", "pixelScale": self.pixel_scale}


@dataclass
class VisionSample:
    image: torch.Tensor  # (3, H, W) uint8
    boxes: torch.Tensor  # (N, 4) float32 in the declared format and coordinates
    labels: torch.Tensor  # (N,) int64 class ids
    masks: torch.Tensor | None  # (N, H, W) bool instance masks (visible pixels), same order as boxes
    keypoints: torch.Tensor | None  # (N, K, 2) float32 in the declared coordinates
    visibility: torch.Tensor | None  # (N, K) int8
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        return int(self.image.shape[-2]), int(self.image.shape[-1])  # (H, W)

    def clone(self) -> "VisionSample":
        c = lambda t: None if t is None else t.clone()
        return VisionSample(self.image.clone(), self.boxes.clone(), self.labels.clone(), c(self.masks), c(self.keypoints), c(self.visibility), dict(self.meta))


def to_xyxy_pixel(boxes: torch.Tensor, fmt: str, coords: str, size_hw: tuple[int, int]) -> torch.Tensor:
    if fmt not in BOX_FORMATS:
        raise ContractError("E_VISION_BOX_FORMAT", f"unknown box format '{fmt}' (have {BOX_FORMATS})")
    if coords not in COORDS:
        raise ContractError("E_VISION_COORDS", f"unknown coordinate system '{coords}' (have {COORDS})")
    h, w = size_hw
    b = boxes.to(torch.float32)
    if coords == "normalized":
        b = b * torch.tensor([w, h, w, h], dtype=torch.float32)
    if fmt == "xyxy":
        return b.clone()
    if fmt == "xywh":
        return torch.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], 1) if len(b) else b.clone()
    return torch.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1) if len(b) else b.clone()


def from_xyxy_pixel(xyxy: torch.Tensor, fmt: str, coords: str, size_hw: tuple[int, int]) -> torch.Tensor:
    h, w = size_hw
    b = xyxy.to(torch.float32)
    if len(b):
        if fmt == "xywh":
            b = torch.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
        elif fmt == "cxcywh":
            b = torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
    if coords == "normalized":
        b = b / torch.tensor([w, h, w, h], dtype=torch.float32)
    return b


def convert_boxes(boxes: torch.Tensor, src: tuple[str, str], dst: tuple[str, str], size_hw: tuple[int, int]) -> torch.Tensor:
    """(format, coords) -> (format, coords). Goes through xyxy pixel coordinates."""
    return from_xyxy_pixel(to_xyxy_pixel(boxes, src[0], src[1], size_hw), dst[0], dst[1], size_hw)


def keypoints_to_pixel(k: torch.Tensor | None, coords: str, size_hw: tuple[int, int]) -> torch.Tensor | None:
    if k is None or coords == "pixel":
        return None if k is None else k.clone()
    h, w = size_hw
    return k * torch.tensor([w, h], dtype=torch.float32)


def keypoints_from_pixel(k: torch.Tensor | None, coords: str, size_hw: tuple[int, int]) -> torch.Tensor | None:
    if k is None or coords == "pixel":
        return None if k is None else k.clone()
    h, w = size_hw
    return k / torch.tensor([w, h], dtype=torch.float32)


def tight_box(mask: torch.Tensor) -> tuple[float, float, float, float] | None:
    """Pixel-edge xyxy box (x2/y2 exclusive) of a boolean mask; None when empty."""
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0


def validate_sample(s: VisionSample, spec: VisionSpec) -> None:
    """Structural contract checks (raises ContractError with a stable code)."""
    h, w = s.size
    n = len(s.boxes)
    if s.image.ndim != 3 or s.image.shape[0] != 3:
        raise ContractError("E_VISION_IMAGE_SHAPE", f"image must be (3, H, W); got {tuple(s.image.shape)}")
    if s.boxes.ndim != 2 or s.boxes.shape[-1] != 4:
        raise ContractError("E_VISION_BOX_SHAPE", f"boxes must be (N, 4); got {tuple(s.boxes.shape)}")
    if len(s.labels) != n:
        raise ContractError("E_VISION_LABEL_COUNT", f"{n} boxes but {len(s.labels)} labels")
    if s.masks is not None and (s.masks.shape[0] != n or tuple(s.masks.shape[-2:]) != (h, w)):
        raise ContractError("E_VISION_MASK_SHAPE", f"masks must be ({n}, {h}, {w}); got {tuple(s.masks.shape)}")
    if s.keypoints is not None and (s.keypoints.shape[0] != n or s.visibility is None or s.visibility.shape != s.keypoints.shape[:2]):
        raise ContractError("E_VISION_KEYPOINT_SHAPE", "keypoints must be (N, K, 2) with visibility (N, K)")
    if n and (int(s.labels.min()) < 0 or int(s.labels.max()) >= len(spec.classes)):
        raise ContractError("E_VISION_LABEL_RANGE", f"labels must be in [0, {len(spec.classes) - 1}]")
    xyxy = to_xyxy_pixel(s.boxes, spec.box_format, spec.coords, (h, w))
    if n and bool(((xyxy[:, 2] <= xyxy[:, 0]) | (xyxy[:, 3] <= xyxy[:, 1])).any()):
        raise ContractError("E_VISION_BOX_DEGENERATE", f"a box has no area in {spec.box_format}/{spec.coords}; the declared format may not match the values")
    for pair in spec.flip_pairs:
        if s.keypoints is not None and max(pair) >= s.keypoints.shape[1]:
            raise ContractError("E_VISION_FLIP_PAIRS", f"flip pair {pair} refers to a keypoint index >= {s.keypoints.shape[1]}")


def consistency(s: VisionSample, spec: VisionSpec, tol_px: float = 0.0) -> dict[str, Any]:
    """Do boxes, masks and keypoints still describe the same objects as the image? Boxes vs the tight bounds of the instance mask, visible keypoints vs the mask."""
    h, w = s.size
    xyxy = to_xyxy_pixel(s.boxes, spec.box_format, spec.coords, (h, w))
    kp = keypoints_to_pixel(s.keypoints, spec.coords, (h, w))
    worst, kp_out, kp_off_mask, empty = 0.0, 0, 0, 0
    for i in range(len(xyxy)):
        if s.masks is not None:
            tb = tight_box(s.masks[i])
            if tb is None:
                empty += 1
            else:
                worst = max(worst, float(torch.abs(torch.tensor(tb) - xyxy[i]).max()))
        if kp is not None:
            for j in range(kp.shape[1]):
                x, y = float(kp[i, j, 0]), float(kp[i, j, 1])
                inside = 0 <= x < w and 0 <= y < h
                v = int(s.visibility[i, j])
                if v > 0 and not inside:
                    kp_out += 1
                if v == 2 and inside and s.masks is not None and not bool(s.masks[i, int(y), int(x)]):
                    kp_off_mask += 1
    return {"instances": len(xyxy), "maxBoxVsMaskPx": worst if s.masks is not None else None, "emptyMasks": empty, "keypointsOutsideCanvasButLabelled": kp_out,
            "visibleKeypointsOffMask": kp_off_mask, "ok": worst <= tol_px and empty == 0 and kp_out == 0 and kp_off_mask == 0}


# ------------------------------------------------------------------------------------------------ loading the SYNTHETIC fixture
def load_npz(path, n: int | None, spec_override: dict[str, Any] | None = None) -> tuple[list[VisionSample], VisionSpec, dict[str, Any]]:
    import json

    d = np.load(path, allow_pickle=False)
    raw = json.loads(str(d["spec"]))
    spec = VisionSpec(classes=raw["classes"], keypoint_names=raw["keypointNames"], flip_pairs=raw["flipPairs"])
    k_total = len(d["images"]) if n is None else min(n, len(d["images"]))
    out = []
    for i in range(k_total):
        img = torch.from_numpy(d["images"][i]).permute(2, 0, 1).contiguous()
        labels_all = d["labels"][i]
        keep = [j for j in range(labels_all.shape[0]) if labels_all[j] >= 0]
        boxes = torch.from_numpy(d["boxes"][i][keep]).float()
        labels = torch.from_numpy(labels_all[keep].astype(np.int64))
        if "masks" in d.files:
            masks = torch.from_numpy(d["masks"][i][keep].astype(bool))
        else:
            inst = torch.from_numpy(d["instances"][i].astype(np.int64))
            masks = torch.stack([inst == (j + 1) for j in range(len(keep))]) if keep else torch.zeros(0, *inst.shape, dtype=torch.bool)
        kps = torch.from_numpy(d["keypoints"][i][keep]).float()
        vis = torch.from_numpy(d["visibility"][i][keep].astype(np.int8))
        out.append(VisionSample(img, boxes, labels, masks, kps, vis, {"id": raw["ids"][i], "synthetic": raw.get("synthetic"), "index": i,
                                                                "imported": raw.get("sampleMetadata", [None]*k_total)[i]}))
    return out, spec, raw
