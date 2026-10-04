"""Shared contract of the `domain` graph kind: wire kinds, runtime values, the operation base class, fixture loading and PNG encoding for inspection records.

Runtime values are `tabular.core.Plain` objects: `data` is the JSON contract that is recorded as the node's output artifact (it describes the value, it is not the tensors),
`obj` is the live Python object consumed by the next node. A wire of one kind never means another: image data, token data and audio data are different kinds."""
from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from PIL import Image

from tabular.core import ExecutionError, Plain, TabularOperation, VType, resolve_path

# wire kinds (edge.kind)
IMAGE_DATA = "image_batch+annotations"
TEXT_CORPUS = "text_corpus"
TOKEN_BATCH = "token_batch"
AUDIO_BATCH = "audio_batch"
FEATURE_BATCH = "feature_batch"
VISION_REPORT, NLP_REPORT, SPEECH_REPORT = "vision_report", "nlp_report", "speech_report"
DOMAIN_KINDS = (IMAGE_DATA, TEXT_CORPUS, TOKEN_BATCH, AUDIO_BATCH, FEATURE_BATCH, VISION_REPORT, NLP_REPORT, SPEECH_REPORT)


class DomainOperation(TabularOperation):
    graph_kind = "domain"
    backend = "python"
    stores_output = True

    def param_count(self, cfg, ins):
        return 0


def value(kind: str, data: dict[str, Any], obj: Any = None) -> Plain:
    return Plain(kind, data, obj)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fixture_path(path: str, command: str = "python examples/make_domain_fixtures.py") -> Path:
    p = resolve_path(path)
    if not p.exists():
        raise ExecutionError("E_FIXTURE_MISSING", f"fixture file '{path}' does not exist. Generate the SYNTHETIC fixtures with `{command}` (deterministic; byte-identical on every run).")
    return p


def png_b64(arr: np.ndarray, palette: list[tuple[int, int, int]] | None = None) -> str:
    """PNG of an HxWx3 uint8 image, or of an HxW label map with an embedded palette (so the browser shows class colours without any client logic)."""
    if arr.ndim == 2:
        im = Image.fromarray(arr.astype(np.uint8), mode="P")
        pal = list(palette or [])
        flat = [c for rgb in pal for c in rgb] + [0] * (768 - 3 * len(pal))
        im.putpalette(flat[:768])
    else:
        im = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode()


def r(x: Any, nd: int = 4) -> Any:
    """Round floats for recorded JSON (full precision is kept in the computation, only the record is rounded)."""
    return None if x is None else round(float(x), nd)


def peek_spec(path: str) -> dict[str, Any] | None:
    """The `spec` JSON stored inside a fixture .npz (cheap: only that member is read). None when the file is missing or unreadable: static checks that need it are then skipped."""
    import json

    try:
        p = resolve_path(path)
        if not p.is_file():
            return None
        with np.load(p, allow_pickle=False) as d:
            return json.loads(str(d["spec"]))
    except Exception:  # noqa: BLE001
        return None
