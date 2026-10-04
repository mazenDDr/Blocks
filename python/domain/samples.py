"""Builders of the domain example graphs (graph kind `domain`), shared by the examples generator and the tests."""
from __future__ import annotations

from typing import Any


def _n(i: str, t: str, cfg: dict[str, Any]) -> dict[str, Any]:
    return {"id": i, "type": t, "version": "1.0.0", "config": cfg}


def _e(k: int, a: tuple[str, str], b: tuple[str, str], kind: str) -> dict[str, Any]:
    return {"id": f"e{k}", "kind": kind, "from": {"node": a[0], "port": a[1]}, "to": {"node": b[0], "port": b[1]}}


def graph(nodes, edges) -> dict[str, Any]:
    return {"schemaVersion": "1.0.0", "graphKind": "domain", "backend": "python", "nodes": nodes, "edges": edges}


def vision_graph(path: str = "examples/data/domain/synthetic_shapes_det.npz", epochs: int = 12) -> dict[str, Any]:
    nodes = [_n("images", "domain.vision_source", {"path": path}),
             _n("augment", "domain.vision_transform", {"steps": [{"op": "random_hflip", "p": 0.5, "seed": 1}, {"op": "pad", "pad": [8, 8, 8, 8], "fill": 128}], "policy": {"clip": True, "min_visible_fraction": 0.0, "min_size": 1.0}}),
             _n("segmenter", "domain.vision_segmenter", {"epochs": epochs})]
    k = "image_batch+annotations"
    return graph(nodes, [_e(1, ("images", "data"), ("augment", "data"), k), _e(2, ("augment", "data"), ("segmenter", "data"), k)])


def nlp_graph(path: str = "examples/fixtures/synthetic_ner.jsonl", epochs: int = 25) -> dict[str, Any]:
    nodes = [_n("corpus", "domain.nlp_source", {"path": path}), _n("tokenize", "domain.nlp_tokenizer", {}), _n("tagger", "domain.nlp_tagger", {"epochs": epochs})]
    return graph(nodes, [_e(1, ("corpus", "corpus"), ("tokenize", "corpus"), "text_corpus"), _e(2, ("tokenize", "tokens"), ("tagger", "tokens"), "token_batch")])


def speech_graph(path: str = "examples/data/domain/synthetic_tones.npz", epochs: int = 60) -> dict[str, Any]:
    nodes = [_n("audio", "domain.audio_source", {"path": path}), _n("features", "domain.audio_features", {}), _n("ctc", "domain.speech_ctc", {"epochs": epochs})]
    return graph(nodes, [_e(1, ("audio", "audio"), ("features", "audio"), "audio_batch"), _e(2, ("features", "features"), ("ctc", "features"), "feature_batch")])
