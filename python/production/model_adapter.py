"""Model-graph (PyTorch CNN image classifier) family for the local production registry (ADR 0018).

A version pins one completed image-classification run of a model graph: its stored graph, the latest *complete* checkpoint (native
state dict, loaded with weights_only), the training preprocessing (RGB, bilinear resize to the graph's input H×W, scaled to [-1, 1]),
the class names from the recorded split, and the native environment and implementation hashes. Registration re-derives the image
folder's identity and requires it to equal the dataset_sha256 recorded at training. It then freezes up to 64 validation images and
their labels into the content-addressed store as the monitoring/warmup reference. Serving never reads the source folder."""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import platform
import threading
import time
from importlib.metadata import version as dist_version
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from tabular.core import dumps
from .pipeline import ProductionError, read_verified

MAX_BATCH = 32
MAX_SIDE = 1024
MAX_REFERENCE = 64
_PY = Path(__file__).resolve().parents[1]
IMPLEMENTATION_FILES = ("production/model_adapter.py", "graph_core/lower.py", "graph_core/validate.py", "operations/layers.py", "operations/core.py",
                        "operations/_common.py", "worker/dataset.py")


def environment() -> dict[str, str]:
    return {"python": platform.python_version(), "torch": torch.__version__, "pillow": dist_version("pillow"), "numpy": dist_version("numpy")}


def implementation() -> dict[str, str]:
    return {f: hashlib.sha256((_PY / f).read_bytes()).hexdigest() for f in IMPLEMENTATION_FILES}


def _graph(store, run_id):
    from graph_core.schema import Graph

    gs = store.artifacts(run_id, "graph")
    if not gs:
        raise ProductionError("E_PIPELINE_NOT_RECORDED", "The run has no stored graph.")
    return Graph.model_validate(json.loads(read_verified(store, gs[0]["sha256"]))), gs[0]["sha256"]


def output_node(graph) -> str | None:
    from graph_core.lower import lower_graph

    try:
        m = lower_graph(graph)
    except Exception:  # noqa: BLE001  (not a lowerable model graph)
        return None
    return m.output_ids[0] if len(m.input_ids) == 1 and len(m.output_ids) == 1 else None


_CANDIDATES: dict[tuple[str, str], str | None] = {}


def is_candidate(store, row) -> str | None:
    """The output node of a completed image-classification model-graph run with a complete checkpoint, else None.
    Memoized per completed run (its records are immutable), because the production overview is polled."""
    key = (str(store.root), row["id"])
    if row["status"] == "completed" and key in _CANDIDATES:
        return _CANDIDATES[key]
    out = _candidate(store, row)
    if row["status"] == "completed":
        _CANDIDATES[key] = out
    return out


def _candidate(store, row) -> str | None:
    if row["status"] != "completed" or row["config"].get("kind") not in (None, "model") or not store.artifacts(row["id"], "split"):
        return None
    if not any(c["status"] == "complete" for c in store.artifacts(row["id"], "checkpoint")):
        return None
    try:
        g, _ = _graph(store, row["id"])
    except ProductionError:
        return None
    return output_node(g) if g.graphKind == "model" else None


def build_manifest(store, run_id: str, node: str) -> dict[str, Any]:
    """Pin the run and freeze the reference. Refuses when the source image folder no longer matches the recorded dataset identity."""
    from graph_core.validate import validate
    from worker.dataset import load_image_folder

    row = store.get_run(run_id)
    if row is None or is_candidate(store, row) != node:
        raise ProductionError("E_PIPELINE_NOT_RECORDED", "Registration needs a completed model-graph image classification run, its output node and a complete checkpoint.")
    g, graph_sha = _graph(store, run_id)
    ck = max((c for c in store.artifacts(run_id, "checkpoint") if c["status"] == "complete"), key=lambda c: (c["step"] or 0, c["id"]))
    split = json.loads(read_verified(store, store.artifacts(run_id, "split")[0]["sha256"]))
    report = validate(g)
    it = report.output_types[sorted(n.id for n in g.nodes if n.type == "core.tensor_input")[0]]["value"]
    h, w = int(it.shape[2]), int(it.shape[3])
    try:
        data = load_image_folder(split["root"], h, w, split["split_seed"], split["val_fraction"])
    except (OSError, ValueError) as e:
        raise ProductionError("E_REFERENCE_SOURCE", f"The training image folder is unavailable ({e}); the reference cannot be frozen.", 409) from e
    if data.files_sha256 != split["dataset_sha256"] or data.val_files != split["val"]:
        raise ProductionError("E_REFERENCE_SOURCE", "The image folder no longer matches the dataset identity recorded at training; registration refused.", 409)
    root = Path(split["root"])
    refs = []
    for rel, label in list(zip(split["val"], split["val_labels"]))[:MAX_REFERENCE]:
        raw = (root / rel).read_bytes()
        refs.append({"file": rel, "label": split["classes"][label], "sha256": store.put_bytes(raw)})
    reference_sha = store.put_bytes(dumps(refs).encode())
    events = store.events(run_id, types=("epoch_end",))
    return {"adapter": "native-pytorch-model-graph-local", "family": "image_classifier", "runId": run_id, "node": node, "graphHash": row["graph_hash"],
            "graphSha256": graph_sha, "modelSha256": ck["sha256"], "checkpointSha256": ck["sha256"], "checkpointStep": ck["step"],
            "inputSize": [h, w], "preprocessing": f"decode image, convert to RGB, bilinear resize to {h}x{w}, scale to [-1, 1] (the training transform)",
            "outputSchema": {"task": "classification", "classes": split["classes"], "target": None},
            "source": {"root": split["root"], "datasetSha256": split["dataset_sha256"], "splitSeed": split["split_seed"], "valFraction": split["val_fraction"]},
            "referenceSha256": reference_sha, "referenceRows": len(refs), "referencePolicy": f"first {MAX_REFERENCE} validation images in recorded order, frozen at registration",
            "environment": environment(), "implementation": implementation(), "fitArtifacts": {},
            "evaluationArtifacts": [], "evaluation": events[-1]["data"] if events else None,
            "inputContract": f"records: [{{imagePng: base64 PNG or JPEG, RGB/RGBA/L, at most {MAX_SIDE} px per side}}]; resized to {h}x{w} like training",
            "tokenizer": None, "customCode": "not supported by this adapter"}


class ModelGraphPipeline:
    """Same interface as production.pipeline.Pipeline: manifest, validate_records, predict (+ reference_records/labels)."""

    def __init__(self, store, manifest: dict[str, Any]):
        from graph_core.lower import lower_graph
        from worker.train import load_checkpoint

        self.store, self.manifest = store, manifest
        verify(store, manifest)
        g, _ = _graph(store, manifest["runId"])
        self.model = lower_graph(g)
        state = load_checkpoint(io.BytesIO(read_verified(store, manifest["checkpointSha256"])))
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()
        self.lock = threading.Lock()

    def validate_records(self, records, max_batch=MAX_BATCH):
        if not records or len(records) > min(max_batch, MAX_BATCH) or len(dumps(records).encode()) > 8_000_000:
            raise ProductionError("E_REQUEST_BOUNDS", f"Image requests take 1-{min(max_batch, MAX_BATCH)} records within 8 MB.", 413)
        for r in records:
            if not isinstance(r, dict) or set(r) != {"imagePng"} or not isinstance(r["imagePng"], str):
                raise ProductionError("E_REQUEST_SCHEMA", "Each record is exactly {imagePng: base64 image}.")

    def _tensor(self, b64: str) -> torch.Tensor:
        from worker.dataset import preprocess_image

        try:
            raw = base64.b64decode(b64.split(",", 1)[-1], validate=True)
            with Image.open(io.BytesIO(raw)) as im:
                if im.format not in ("PNG", "JPEG") or im.mode not in ("RGB", "RGBA", "L") or max(im.size) > MAX_SIDE:
                    raise ProductionError("E_REQUEST_SCHEMA", f"Images must be PNG/JPEG, RGB/RGBA/L, at most {MAX_SIDE} px per side.")
                h, w = self.manifest["inputSize"]
                return preprocess_image(im, h, w)
        except ProductionError:
            raise
        except (binascii.Error, OSError, ValueError) as e:
            raise ProductionError("E_REQUEST_SCHEMA", f"Could not decode the image: {e}") from e

    def predict(self, records):
        t0 = time.perf_counter()
        x = torch.stack([self._tensor(r["imagePng"]) for r in records])
        t1 = time.perf_counter()
        with self.lock, torch.no_grad():
            logits = self.model(x)
        t2 = time.perf_counter()
        probs = torch.softmax(logits, 1)
        classes = self.manifest["outputSchema"]["classes"]
        top = probs.argmax(1).tolist()
        result = {"predictions": [classes[i] for i in top], "probabilities": probs.tolist(), "logits": logits.tolist(), "classes": classes}
        return result, {"preprocessingMs": (t1 - t0) * 1000, "inferenceMs": (t2 - t1) * 1000, "postprocessingMs": (time.perf_counter() - t2) * 1000,
                        "inputShape": list(x.shape)}

    def reference(self) -> list[dict[str, Any]]:
        return json.loads(read_verified(self.store, self.manifest["referenceSha256"]))

    def reference_records(self, n: int | None = None) -> list[dict[str, Any]]:
        return [{"imagePng": base64.b64encode(read_verified(self.store, r["sha256"])).decode()} for r in self.reference()[:n]]


def verify(store, manifest) -> None:
    """Re-checked on every use: environment, implementation, and every pinned artifact's hash."""
    if manifest["environment"] != environment() or manifest["implementation"] != implementation():
        raise ProductionError("E_SERVING_ENVIRONMENT", "Pinned native environment or implementation differs; serving is refused.", 409)
    if not any(c["sha256"] == manifest["checkpointSha256"] and c["status"] == "complete" for c in store.artifacts(manifest["runId"], "checkpoint")):
        raise ProductionError("E_ARTIFACT_TRUST", "The pinned checkpoint is not a complete recorded checkpoint of its run.", 409)
    for sha in (manifest["checkpointSha256"], manifest["graphSha256"], manifest["referenceSha256"]):
        read_verified(store, sha)
