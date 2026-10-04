"""Vision operations of the `domain` graph kind: annotated image source, box conversion, annotation-preserving transforms, tiny segmenter with mAP / IoU."""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from pydantic import Field

from graph_core.registry import register
from graph_core.types import Fix, OpError
from operations._common import StrictConfig
from tabular.core import ExecutionError, VType, resolve_path
from vision import contract as C
from vision import segment as S
from vision import transforms as T

from . import checkpoints as CP
from .core import IMAGE_DATA, VISION_REPORT, DomainOperation, fixture_path, peek_spec, png_b64, r, sha256_file, value
from .core import Plain

DEFAULT_NPZ = "examples/data/domain/synthetic_shapes_det.npz"
PALETTE = S.PALETTE
SYNTH = "SYNTHETIC fixture (examples/make_domain_fixtures.py): drawn shapes, not photographs."


def _info(raw: dict[str, Any] | None, fmt: str, coords: str, n: int | None) -> dict[str, Any]:
    d: dict[str, Any] = {"boxFormat": fmt, "coords": coords, "n": n, "height": None, "width": None, "classes": [], "keypointNames": [], "flipPairs": [], "hasMasks": True,
                         "hasKeypoints": True, "colorSpace": "RGB"}
    if raw:
        d.update(height=raw["size"], width=raw["size"], classes=raw["classes"], keypointNames=raw["keypointNames"], flipPairs=raw["flipPairs"], n=n if n is not None else raw["n"])
    return d


def _spec_of(info: dict[str, Any]) -> C.VisionSpec:
    return C.VisionSpec(info["classes"], info["keypointNames"], info["flipPairs"], info["boxFormat"], info["coords"])


def _record(s: C.VisionSample, spec: C.VisionSpec, nm: str = "") -> dict[str, Any]:
    """One inspectable sample: the image, its boxes in the DECLARED format/coordinates AND in xyxy pixels, instance label map, keypoints with visibility."""
    h, w = s.size
    cm = S.class_map(s).numpy().astype(np.uint8)
    xyxy = C.to_xyxy_pixel(s.boxes, spec.box_format, spec.coords, (h, w))
    kp = C.keypoints_to_pixel(s.keypoints, spec.coords, (h, w))
    return {"id": s.meta.get("id"), "size": [h, w], "image": png_b64(s.image.permute(1, 2, 0).numpy()), "labelMap": png_b64(cm, PALETTE),
            "boxesDeclared": [[r(x, 5) for x in b] for b in s.boxes.tolist()], "boxesXyxyPixel": [[r(x, 3) for x in b] for b in xyxy.tolist()], "labels": s.labels.tolist(),
            "keypoints": None if kp is None else [[[r(x, 3), r(y, 3), int(v)] for (x, y), v in zip(k, vv)] for k, vv in zip(kp, s.visibility)],
            "maskAreas": None if s.masks is None else [int(m.sum()) for m in s.masks]}


def _data_value(samples: list[C.VisionSample], spec: C.VisionSpec, extra: dict[str, Any]) -> Plain:
    h, w = samples[0].size
    data = {"contract": {"n": len(samples), "height": h, "width": w, **spec.to_json(), "hasMasks": samples[0].masks is not None, "hasKeypoints": samples[0].keypoints is not None,
                         "instances": int(sum(len(s.labels) for s in samples))}, **extra}
    return value(IMAGE_DATA, data, {"samples": samples, "spec": spec})


# ================================================================================================ source
class VisionSourceConfig(StrictConfig):
    path: str = Field(DEFAULT_NPZ, description="SYNTHETIC detection fixture (.npz written by examples/make_domain_fixtures.py)")
    n: int | None = Field(None, ge=1, title="use the first n images (empty = all)")
    box_format: Literal["xyxy", "xywh", "cxcywh"] = Field("xyxy", title="declared box format")
    coords: Literal["pixel", "normalized"] = Field("pixel", title="declared coordinates (normalized = fractions of image width/height)")


@register
class VisionSource(DomainOperation):
    type = "domain.vision_source"
    inputs = ()
    outputs = ("data",)
    out_kinds = {"data": IMAGE_DATA}
    Config = VisionSourceConfig
    summary_kind = "vision_data"

    def infer(self, cfg, ins, node_id):
        if not cfg.path:
            raise OpError("E_SOURCE_NOT_SET", "No fixture file chosen.", None, [Fix("Set 'path' to the vision fixture (.npz)")])
        if not resolve_path(cfg.path).is_file():
            raise OpError("E_FIXTURE_MISSING", f"Fixture '{cfg.path}' was not found. Generate the SYNTHETIC fixtures with `python examples/make_domain_fixtures.py`.", None, [Fix("Run python examples/make_domain_fixtures.py")])
        return {"data": VType(IMAGE_DATA, _info(peek_spec(cfg.path), cfg.box_format, cfg.coords, cfg.n))}

    def execute(self, cfg, ins, ctx):
        p = fixture_path(cfg.path)
        samples, spec, raw = C.load_npz(p, cfg.n)
        spec.box_format, spec.coords = cfg.box_format, cfg.coords
        for s in samples:
            s.boxes = C.convert_boxes(s.boxes, ("xyxy", "pixel"), (cfg.box_format, cfg.coords), s.size)
            s.keypoints = C.keypoints_from_pixel(s.keypoints, cfg.coords, s.size)
            try:
                C.validate_sample(s, spec)
            except C.ContractError as e:
                raise ExecutionError(e.code, f"{s.meta.get('id')}: {e.message}") from e
        cons = [C.consistency(s, spec, 0.0) for s in samples]
        hist = np.bincount(np.concatenate([s.labels.numpy() for s in samples] or [np.zeros(0, int)]), minlength=len(spec.classes)).tolist()
        sha = sha256_file(p)
        out = _data_value(samples, spec, {"source": {"path": str(p), "configuredPath": cfg.path, "sha256": sha, "synthetic": True, "note": SYNTH}})
        summary = {"path": str(p), "sha256": sha, "synthetic": True, "note": SYNTH, "contract": out.data["contract"], "classHistogram": dict(zip(spec.classes, hist)),
                   "emptyImages": sum(1 for s in samples if len(s.labels) == 0), "occludedKeypoints": int(sum(int((s.visibility == 1).sum()) for s in samples)),
                   "consistency": {"ok": sum(c["ok"] for c in cons), "total": len(cons), "worstBoxVsMaskPx": max(c["maxBoxVsMaskPx"] for c in cons)},
                   "samples": [_record(s, spec) for s in samples[:6]], "provenance": {"source": "read from the fixture file; boxes converted to the declared format/coordinates for the contract"}}
        return {"data": out}, summary

    def explain(self, cfg, inputs, outputs):
        return {"equation": f"samples = (image CHW uint8, boxes [{cfg.box_format}, {cfg.coords}], labels, instance masks, keypoints (x, y, visibility), metadata)",
                "rule": "Boxes are tight bounds of the VISIBLE pixels with exclusive max edges; keypoint visibility 2 = visible, 1 = occluded, 0 = not labelled. The declared format and coordinates are part of the wire's type.",
                "note": SYNTH}


# ================================================================================================ box conversion
class BoxConvertConfig(StrictConfig):
    to_format: Literal["xyxy", "xywh", "cxcywh"] = "xyxy"
    to_coords: Literal["pixel", "normalized"] = "pixel"


@register
class BoxConvert(DomainOperation):
    type = "domain.vision_box_convert"
    inputs = ("data",)
    outputs = ("data",)
    in_kinds = {"data": IMAGE_DATA}
    out_kinds = {"data": IMAGE_DATA}
    Config = BoxConvertConfig
    summary_kind = "vision_data"

    def infer(self, cfg, ins, node_id):
        return {"data": VType(IMAGE_DATA, {**ins["data"].info, "boxFormat": cfg.to_format, "coords": cfg.to_coords})}

    def execute(self, cfg, ins, ctx):
        d = ins["data"]
        samples, spec = d.obj["samples"], d.obj["spec"]
        new = C.VisionSpec(spec.classes, spec.keypoint_names, spec.flip_pairs, cfg.to_format, cfg.to_coords)
        out = []
        for s in samples:
            c = s.clone()
            c.boxes = C.convert_boxes(s.boxes, (spec.box_format, spec.coords), (cfg.to_format, cfg.to_coords), s.size)
            c.keypoints = C.keypoints_from_pixel(C.keypoints_to_pixel(s.keypoints, spec.coords, s.size), cfg.to_coords, s.size)
            out.append(c)
        v = _data_value(out, new, {"source": d.data.get("source")})
        s0 = out[0]
        return {"data": v}, {"contract": v.data["contract"], "from": [spec.box_format, spec.coords], "to": [cfg.to_format, cfg.to_coords],
                             "exampleBefore": {"boxes": [r(x, 5) for x in samples[0].boxes[0].tolist()] if len(samples[0].boxes) else None},
                             "exampleAfter": {"boxes": [r(x, 5) for x in s0.boxes[0].tolist()] if len(s0.boxes) else None},
                             "samples": [_record(s, new) for s in out[:4]], "provenance": {"source": "converted through xyxy pixel coordinates"}}

    def explain(self, cfg, inputs, outputs):
        return {"equation": "xyxy_pixel = to_xyxy(boxes, format, coords, image size);  boxes' = from_xyxy(xyxy_pixel, to_format, to_coords)",
                "rule": "xywh = (x, y, w, h); cxcywh = (cx, cy, w, h); normalized divides x by the image width and y by the image height. Keypoints follow the same coordinate system."}


# ================================================================================================ transform
class VisionTransformConfig(StrictConfig):
    steps: list[T.Step] = Field(default_factory=lambda: [T.Step(op="hflip"), T.Step(op="resize", size=[64, 64])], title="geometric steps, applied in order")
    policy: T.BoxPolicy = Field(default_factory=T.BoxPolicy, title="box policy (applied after every step)")
    n_examples: int = Field(4, ge=1, le=12, title="before/after examples recorded")


@register
class VisionTransform(DomainOperation):
    type = "domain.vision_transform"
    inputs = ("data",)
    outputs = ("data",)
    in_kinds = {"data": IMAGE_DATA}
    out_kinds = {"data": IMAGE_DATA}
    Config = VisionTransformConfig
    summary_kind = "vision_transform"

    def infer(self, cfg, ins, node_id):
        info = dict(ins["data"].info)
        known = bool(info.get("height"))
        try:
            h, w = T.static_size(cfg.steps, (info["height"], info["width"]) if known else (10 ** 6, 10 ** 6))
        except C.ContractError as e:
            raise OpError(e.code, e.message, "data", [Fix("Change the step parameters so the window fits the canvas")])
        if known:
            info["height"], info["width"] = h, w
        for st in cfg.steps:
            if st.op == "pad" and st.fill not in range(256):
                raise OpError("E_VISION_TRANSFORM_PARAM", "pad fill must be 0-255")
        return {"data": VType(IMAGE_DATA, info)}

    def warnings(self, cfg, ins, node_id):
        info, w = ins["data"].info, []
        flips = any(s.op in ("hflip", "random_hflip") for s in cfg.steps)
        if info.get("hasKeypoints") and flips and not info.get("flipPairs"):
            w.append(("W_VISION_FLIP_PAIRS", "Keypoints are present and a horizontal flip is requested, but no left/right flip pairs are declared: left and right keypoints will NOT be exchanged.", "data"))
        if not cfg.policy.clip:
            w.append(("W_VISION_NO_CLIP", "Boxes are not clipped to the canvas: a box may extend beyond the image.", "data"))
        if any(s.op == "affine" for s in cfg.steps):
            w.append(("W_VISION_AFFINE_BOX", "After an affine step a box is the bounding box of the transformed corners; it can be looser than the tight bounds of the transformed mask.", "data"))
        return w

    def execute(self, cfg, ins, ctx):
        d = ins["data"]
        samples, spec = d.obj["samples"], d.obj["spec"]
        out, logs, cons_before, cons_after = [], [], [], []
        for i, s in enumerate(samples):
            try:
                o, lg = T.apply_pipeline(s, spec, cfg.steps, cfg.policy, i)
            except C.ContractError as e:
                raise ExecutionError(e.code, f"{s.meta.get('id')}: {e.message}") from e
            out.append(o)
            logs.append(lg)
            cons_after.append(C.consistency(o, spec, 0.0))
            cons_before.append(C.consistency(s, spec, 0.0))
        removed = sum(len(x["removed"]) for lg in logs for x in lg)
        v = _data_value(out, spec, {"source": d.data.get("source"), "transforms": [st.model_dump(exclude_defaults=True) for st in cfg.steps], "policy": cfg.policy.model_dump()})
        v.data["contract"]["removedInstances"] = removed
        ex = []
        for i in range(min(cfg.n_examples, len(samples))):
            ex.append({"before": _record(samples[i], spec), "after": _record(out[i], spec), "log": logs[i]})
        summary = {"contract": v.data["contract"], "contractBefore": d.data["contract"], "steps": [st.model_dump(exclude_defaults=True) for st in cfg.steps], "policy": cfg.policy.model_dump(),
                   "removedInstances": removed, "instancesBefore": d.data["contract"]["instances"], "instancesAfter": v.data["contract"]["instances"],
                   "consistencyAfter": {"ok": sum(c["ok"] for c in cons_after), "total": len(cons_after), "worstBoxVsMaskPx": max((c["maxBoxVsMaskPx"] or 0.0) for c in cons_after),
                                         "keypointsOutside": sum(c["keypointsOutsideCanvasButLabelled"] for c in cons_after), "visibleKeypointsOffMask": sum(c["visibleKeypointsOffMask"] for c in cons_after)},
                   "examples": ex, "provenance": {"source": "torchvision.transforms.v2.functional on tv_tensors; box policy and keypoint flip pairs applied by workbench code; parameters are the ACTUAL ones used"}}
        return {"data": v}, summary

    def explain(self, cfg, inputs, outputs):
        return {"equation": " -> ".join(s.op for s in cfg.steps) or "(no steps)",
                "rule": "Each step transforms image, instance masks, boxes and keypoints together (torchvision tv_tensors). Afterwards the box policy clips boxes, removes instances whose visible fraction or size falls below the thresholds "
                        "(box, label, mask and keypoints are removed together) and marks keypoints outside the canvas invisible. A horizontal flip exchanges the declared keypoint flip pairs.",
                "note": "Keypoint coordinates are continuous (pixel i spans [i, i+1)). torchvision's own KeyPoints flip uses integer pixel centres, so flips are computed explicitly here."}


# ================================================================================================ segmenter
class SegmenterConfig(StrictConfig):
    resume_model_id: str | None = Field(None, pattern=r"^[0-9a-f]{64}$", description="Internal model manifest identity to resume at a completed epoch; epochs is the total target.")
    epochs: int = Field(12, ge=1, le=200)
    lr: float = Field(3e-3, gt=0)
    width: int = Field(12, ge=2, le=64, title="base channels")
    batch_size: int = Field(16, ge=1, le=256)
    val_fraction: float = Field(0.25, gt=0, lt=1)
    seed: int = 0
    n_inspect: int = Field(8, ge=1, le=24, title="validation samples recorded for the overlay inspector")


@register
class Segmenter(DomainOperation):
    type = "domain.vision_segmenter"
    inputs = ("data",)
    outputs = ("report",)
    in_kinds = {"data": IMAGE_DATA}
    out_kinds = {"report": VISION_REPORT}
    Config = SegmenterConfig
    summary_kind = "vision_seg"

    def infer(self, cfg, ins, node_id):
        info = ins["data"].info
        if (info["boxFormat"], info["coords"]) != ("xyxy", "pixel"):
            raise OpError("E_VISION_BOX_FORMAT", f"The segmenter's evaluation (torchmetrics mAP, IoU against masks) needs boxes as xyxy in pixel coordinates, but the wire declares "
                          f"{info['boxFormat']} / {info['coords']}. Boxes in another format would be read as different numbers.", "data",
                          [Fix("Insert a 'Box format conversion' node with to_format=xyxy, to_coords=pixel before this node")])
        if not info.get("hasMasks"):
            raise OpError("E_VISION_NO_MASKS", "The segmenter trains on instance masks; this wire carries none.", "data")
        return {"report": VType(VISION_REPORT, {"classes": info["classes"]})}

    def execute(self, cfg, ins, ctx):
        d = ins["data"]
        samples, spec = d.obj["samples"], d.obj["spec"]
        signature = CP.fingerprint(cfg.model_dump(exclude={"epochs", "n_inspect", "resume_model_id"}), d.data,
                                   [t for s in samples for t in (s.image, S.class_map(s))])
        saved = CP.resume(ctx, cfg.resume_model_id, "vision", signature)
        res = S.train_and_eval(samples, spec, epochs=cfg.epochs, lr=cfg.lr, width=cfg.width, batch_size=cfg.batch_size, seed=cfg.seed, val_fraction=cfg.val_fraction, n_inspect=cfg.n_inspect, resume_state=saved)
        names = ["background"] + spec.classes
        seg = res["seg"]
        recs = []
        for x in res["records"]:
            recs.append({k: v for k, v in x.items() if k not in ("image", "gtMask", "predMask")} | {"size": list(x["image"].shape[:2]), "image": png_b64(x["image"]), "gtMask": png_b64(x["gtMask"], PALETTE),
                                                                                                       "predMask": png_b64(x["predMask"], PALETTE)})
        m = res["det"]
        prov_seg = "confusion matrix over all validation pixels; per-class IoU = TP/(TP+FP+FN), Dice = 2TP/(2TP+FP+FN) (checked against torchmetrics in tests)"
        import torchmetrics

        prov_det = f"torchmetrics {torchmetrics.__version__} MeanAveragePrecision (pycocotools backend, IoU thresholds 0.50:0.95) on boxes of {S.CONNECTIVITY} of the predicted mask, score = mean class probability; DERIVED detections"
        summary = {"classes": names, "nParams": res["nParams"], "config": cfg.model_dump(), "split": {"seed": cfg.seed, "nTrain": res["nTrain"], "nVal": res["nVal"], "valIndices": res["valIdx"]},
                   "curve": res["curve"], "segmentation": {**seg, "classNames": names, "provenance": prov_seg}, "detection": {**m, "provenance": prov_det},
                   "samples": recs, "palette": PALETTE[:len(names)], "seconds": res["seconds"], "synthetic": True, "note": SYNTH,
                   "contract": d.data["contract"], "transformsApplied": d.data.get("transforms", []), "provenance": {"source": d.data.get("source"), "torch": torch.__version__,
                                                                                                             "model": "TinyFCN (2 encoder levels, bilinear upsample, skip connection)"}}
        summary["checkpoint"] = CP.persist(ctx, "vision", signature, res["trainingState"],
            {"architecture": {"n_classes": len(spec.classes) + 1, "width": cfg.width}, "classes": names,
             "size": list(samples[0].size), "input": "post-geometry RGB uint8 image; no implicit resize/crop/random flip",
             "normalization": "(uint8 / 255 - 0.5) / 0.25", "trainingGeometry": d.data.get("transforms"), "policy": d.data.get("policy")},
            d.data.get("source"), cfg.resume_model_id, {"imagePng": recs[0]["image"]})
        data = {"metrics": {"meanIoU": seg["meanIoU"], "meanDice": seg["meanDice"], "pixelAccuracy": seg["pixelAccuracy"], "map": m["map"], "map50": m["map50"]}, "nParams": res["nParams"]}
        return {"report": value(VISION_REPORT, data, None)}, summary

    def explain(self, cfg, inputs, outputs):
        return {"equation": "logits = FCN(image);  loss = cross_entropy(logits, class map);  IoU_c = TP_c / (TP_c + FP_c + FN_c)",
                "rule": "Instance masks become one class map (0 = background, c+1 = class c). Detections are read off the predicted mask as connected components, so mAP here measures what this segmenter's masks imply, not a detector.",
                "note": SYNTH}
