"""Semantic hash: canonical JSON of the spec only. The UI document is never an input."""
from __future__ import annotations

import hashlib
import json

from .schema import Graph


def canonical_json(graph: Graph) -> str:
    data = graph.to_json()
    data["nodes"] = sorted(data["nodes"], key=lambda n: n["id"])
    data["edges"] = sorted(data["edges"], key=lambda e: e["id"])
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def semantic_hash(graph: Graph) -> str:
    return hashlib.sha256(canonical_json(graph).encode("utf-8")).hexdigest()
