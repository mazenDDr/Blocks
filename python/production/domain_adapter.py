"""Domain (vision / NLP / speech) family for the local production registry (ADR 0017).

A version pins one recorded internal domain model manifest (`domain_model` artifact: native state dictionary, preprocessing,
tokenizer/labels/alphabet, environment and implementation hashes; ADR 0014). Releases, routes, admission, traces, idempotency,
cancellation, labels, replay and traffic are the same records as the tabular adapter. Native inference is
`domain.inference.run` on a model rebuilt once from the verified state dictionary; the manifest, environment and implementation
are re-verified on every use. Requests are the same typed records as `/api/domain/models/{id}/predict`; the pinned bounds are
1-4 records per request and 1.5 MB decoded JSON."""
from __future__ import annotations

import base64
import io
import threading
import time
from typing import Any

import numpy as np
from PIL import Image

from tabular.core import ExecutionError, dumps
from .pipeline import ProductionError

MAX_BATCH = 4


def _wrap(fn):
    try:
        return fn()
    except ExecutionError as e:
        status = 409 if e.code.startswith("E_DOMAIN_CHECKPOINT") else 422
        raise ProductionError("E_REQUEST_SCHEMA" if status == 422 and e.code in ("E_DOMAIN_INPUT", "E_AUDIO_SAMPLE_RATE") else e.code, e.message, status) from e


def manifest_of(store, model_id: str) -> dict[str, Any]:
    from domain import checkpoints as CP

    m = _wrap(lambda: CP.read_manifest(store, model_id))
    return {"adapter": "native-domain-local", "family": m["family"], "modelId": model_id, "runId": m["runId"], "node": m["node"],
            "graphHash": m["graphHash"], "graphSha256": next((a["sha256"] for a in store.artifacts(m["runId"], "graph")), None),
            "modelSha256": m["checkpointSha256"], "checkpointSha256": m["checkpointSha256"], "epochs": m["epochs"], "parentModelId": m.get("parentModelId"),
            "exampleSha256": m["exampleSha256"], "source": m["source"], "environment": m["environment"], "implementation": m["implementation"],
            "fitArtifacts": {}, "evaluationArtifacts": [a["sha256"] for a in store.artifacts(m["runId"], "node_summary") if a["meta"].get("node") == m["node"]],
            "outputSchema": {"task": m["family"], "classes": m["inference"].get("classes") or m["inference"].get("labels"), "target": None},
            "inputContract": INPUT_CONTRACT[m["family"]], "tokenizer": "pinned tokenizer JSON in the model manifest" if m["family"] == "nlp" else None,
            "customCode": "not supported by this adapter"}


INPUT_CONTRACT = {
    "vision": "records: [{imagePng: base64 RGB PNG at the pinned post-geometry size}]",
    "nlp": "records: [{text: original text, 1-4000 characters}]",
    "speech": "records: [{samples: [[PCM in -1..1] per channel], sampleRate: pinned rate}]",
}


class DomainPipeline:
    """Same interface as production.pipeline.Pipeline: manifest, validate_records, predict."""

    def __init__(self, store, model_id: str):
        from domain.inference import load

        self.store, self.model_id = store, model_id
        self.manifest = manifest_of(store, model_id)
        self.m, self.state, self.model = _wrap(lambda: load(store, model_id))
        self.lock = threading.Lock()  # one native forward pass at a time per loaded model

    def validate_records(self, records, max_batch=MAX_BATCH):
        if not records or len(records) > min(max_batch, MAX_BATCH) or len(dumps(records).encode()) > 1_500_000:
            raise ProductionError("E_REQUEST_BOUNDS", f"Domain requests take 1-{min(max_batch, MAX_BATCH)} records within 1.5 MB.", 413)

    def predict(self, records):
        from domain.inference import run

        t0 = time.perf_counter()
        with self.lock:
            out = _wrap(lambda: run(self.m, self.state, self.model, self.model_id, records))
        ms = (time.perf_counter() - t0) * 1000
        return {"predictions": out["predictions"], "family": self.m["family"]}, {
            "inferenceMs": ms, "preprocessingMs": None, "postprocessingMs": None,
            "timingNote": "native preprocessing, forward pass and decoding are measured together for domain models"}

    def reference_records(self) -> list[dict[str, Any]]:
        """The recorded SYNTHETIC held-out example(s) of the source run (never training data)."""
        import json

        from domain import checkpoints as CP

        sha = self.m.get("exampleSha256")
        if not sha or not any(a["kind"] == "domain_inference_example" and a["sha256"] == sha and a["meta"].get("node") == self.m["node"]
                              for a in self.store.artifacts(self.m["runId"])):
            raise ProductionError("E_REFERENCE_MISSING", "This model has no recorded held-out example; warmup and monitoring reference are unavailable.", 409)
        return [json.loads(_wrap(lambda: CP.verified(self.store, sha)))]


# ------------------------------------------------------------------------------------------------ monitoring helpers
def input_features(family: str, record: dict[str, Any]) -> dict[str, float]:
    """Scalar descriptors of one captured input (descriptive drift evidence only)."""
    if family == "nlp":
        t = record["text"]
        return {"characters": float(len(t)), "words": float(len(t.split()))}
    if family == "speech":
        x = np.asarray(record["samples"], dtype=np.float64)
        return {"durationSeconds": x.shape[1] / record["sampleRate"], "rms": float(np.sqrt((x ** 2).mean()))}
    with Image.open(io.BytesIO(base64.b64decode(record["imagePng"]))) as im:
        a = np.asarray(im, dtype=np.float64)
    return {"meanIntensity": float(a.mean()), "intensityStd": float(a.std())}


def prediction_values(family: str, prediction: dict[str, Any]) -> list[Any]:
    """Categorical values whose frequencies describe a prediction: word labels, decoded characters, or mask pixel classes."""
    if family == "nlp":
        return [w["label"] for w in prediction["words"]]
    if family == "speech":
        return list(prediction["text"]) or ["<empty>"]
    return mask_of(prediction).reshape(-1).tolist()


def mask_of(prediction: dict[str, Any]) -> np.ndarray:
    with Image.open(io.BytesIO(base64.b64decode(prediction["maskPng"]))) as im:
        return np.asarray(im)  # palette PNG: pixel values are class indices


def validate_labels(family: str, predictions: list[dict[str, Any]], labels: list[Any], classes: list[str] | None) -> None:
    if len(labels) != len(predictions):
        raise ProductionError("E_LABEL_ALIGNMENT", "Give one label per predicted record.")
    for p, y in zip(predictions, labels):
        if family == "nlp":
            ok = isinstance(y, list) and len(y) == len(p["words"]) and all(isinstance(v, str) and v in (classes or []) for v in y)
            msg = "NLP labels are one IOB2 label per original word, from the pinned label set."
        elif family == "speech":
            ok, msg = isinstance(y, str), "Speech labels are reference transcripts (strings)."
        else:
            h, w = p["size"]
            ok = (isinstance(y, list) and len(y) == h and all(isinstance(r, list) and len(r) == w and all(type(v) is int and 0 <= v <= len(classes or []) for v in r) for r in y))
            msg = f"Vision labels are {h}x{w} integer class masks (0 = background, 1..{len(classes or [])} = pinned classes)."
        if not ok:
            raise ProductionError("E_LABEL_SCHEMA", msg)


def quality(family: str, pairs: list[tuple[dict[str, Any], Any]], n_classes: int) -> dict[str, Any]:
    """Label-based quality on explicitly supplied ground truth, with the family's native metric conventions."""
    if family == "nlp":
        from seqeval.metrics import f1_score

        truth = [y for _, y in pairs]
        pred = [[w["label"] for w in p["words"]] for p, _ in pairs]
        flat = [(a == b) for t, q in zip(truth, pred) for a, b in zip(t, q)]
        spans = any(v != "O" for seq in truth + pred for v in seq)
        # With no entity span in the labels or the predictions, span precision/recall are 0/0: report undefined, not seqeval's 0.
        return {"wordAccuracy": float(np.mean(flat)) if flat else None, "spanMicroF1": float(f1_score(truth, pred)) if spans else None,
                "spanMicroF1Note": None if spans else "undefined: no entity spans in the supplied labels or the predictions",
                "method": "word accuracy; seqeval default (CoNLL) span F1"}
    if family == "speech":
        from speech.ctc import corpus_rates

        r = corpus_rates([(y, p["text"]) for p, y in pairs])
        return {"cer": r["cer"]["rate"], "wer": r["wer"]["rate"], "method": "corpus edit-distance rates (total edits / total reference length)"}
    import torch

    from vision.segment import confusion, seg_metrics

    conf = sum(confusion(torch.from_numpy(mask_of(p).astype(np.int64)), torch.tensor(y, dtype=torch.int64), n_classes + 1) for p, y in pairs)
    s = seg_metrics(conf)
    return {"pixelAccuracy": s["pixelAccuracy"], "meanIoU": s["meanIoU"], "meanDice": s["meanDice"], "method": "confusion-matrix IoU/Dice over present classes"}
