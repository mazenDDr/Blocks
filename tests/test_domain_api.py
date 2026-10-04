"""A56–A58: typed-domain graphs, contract refusals and actual worker/inspection journeys."""
import hashlib
import json

import pytest

from fastapi.testclient import TestClient
from control.app import create_app
from conftest import EXAMPLES
from domain import samples
from graph_core import registry
from graph_core.schema import Graph
from graph_core.validate import validate
from tabular_helpers import set_cfg
from test_tabular_api import submit, wait_for

BUILDERS = [samples.vision_graph, samples.nlp_graph, samples.speech_graph]
DOMAIN_OPS = {
    "domain.vision_source", "domain.vision_box_convert", "domain.vision_transform", "domain.vision_segmenter",
    "domain.nlp_source", "domain.nlp_tokenizer", "domain.nlp_tagger", "domain.audio_source",
    "domain.audio_resample", "domain.audio_features", "domain.speech_ctc",
}


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "wb")) as c:
        yield c


def test_all_domain_operations_have_typed_ports_explain_and_fast_graphs(domain_fixtures):
    ops = [o for o in registry.all_ops() if o.graph_kind == "domain"]
    assert {o.type for o in ops} == DOMAIN_OPS
    for op in ops:
        assert set(op.in_kinds) == set(op.inputs) and set(op.out_kinds) == set(op.outputs)
        assert op.summary_kind and op.explain(op.Config(), {}, {})["equation"]
    for build in BUILDERS:
        r = validate(Graph.model_validate(build(epochs=1)))
        assert r.ok, [(d.code, d.message) for d in r.errors]


@pytest.mark.parametrize("build,node,patch,code", [
    (samples.vision_graph, "images", {"box_format": "xywh"}, "E_VISION_BOX_FORMAT"),
    (samples.vision_graph, "images", {"coords": "normalized"}, "E_VISION_BOX_FORMAT"),
    (samples.nlp_graph, "tagger", {"label_policy": "all_subwords"}, "E_NLP_LABEL_POLICY"),
    (samples.nlp_graph, "tagger", {"loss_ignore_index": -1}, "E_NLP_IGNORE_INDEX"),
    (samples.speech_graph, "features", {"expected_sample_rate": 8000}, "E_AUDIO_SAMPLE_RATE"),
    (samples.speech_graph, "features", {"win_length": 401}, "E_AUDIO_FRAME_CONFIG"),
])
def test_contract_refusals_precede_worker_submission(client, domain_fixtures, build, node, patch, code):
    c = client
    g = Graph.model_validate(build(epochs=1))
    set_cfg(g, node, **patch)
    r = c.post("/api/validate", json={"graph": g.to_json()}).json()
    assert not r["ok"] and any(d["code"] == code and d["nodeId"] == ("segmenter" if node == "images" else node) for d in r["diagnostics"])
    refused = submit(c, g)
    assert refused.status_code == 422 and refused.json()["detail"]["code"] == "execution_blocked"
    assert c.get("/api/runs").json()["runs"] == []


def test_wire_kind_cannot_alias_other_domain_or_tensor_kinds(domain_fixtures):
    g = Graph.model_validate(samples.speech_graph(epochs=1))
    for kind in ["tensor", "token_batch", "image_batch+annotations"]:
        g.edges[0].kind = kind
        assert "E_EDGE_KIND" in {d.code for d in validate(g).errors}


def test_explicit_box_conversion_restores_a_valid_contract(domain_fixtures):
    g = samples.vision_graph(epochs=1)
    g["nodes"][0]["config"]["box_format"] = "cxcywh"
    g["nodes"].insert(2, {"id": "convert", "type": "domain.vision_box_convert", "config": {}, "version": "1.0.0"})
    g["edges"][1]["to"]["node"] = "convert"
    g["edges"].append({"id": "e3", "kind": "image_batch+annotations", "from": {"node": "convert", "port": "data"}, "to": {"node": "segmenter", "port": "data"}})
    assert validate(Graph.model_validate(g)).ok


@pytest.mark.parametrize("build,last,expected", [
    (samples.vision_graph, "segmenter", "segmentation"),
    (samples.nlp_graph, "tagger", "spanMetrics"),
    (samples.speech_graph, "ctc", "rates"),
])
def test_actual_domain_worker_inspection_is_recorded_and_read_only(client, domain_fixtures, build, last, expected):
    c = client
    g = Graph.model_validate(build(epochs=2))
    # Bound integration training; native-learning quality has its own domain tests.
    g.nodes[0].config["n"] = 24
    r = submit(c, g)
    assert r.status_code == 201, r.text
    rid = r.json()["runId"]
    final = wait_for(c, rid)
    assert final["status"] == "completed", final
    assert final["kind"] == "domain" and final["progress"] == {"nodesDone": 3, "nodes": 3}
    assert final["libraries"]["torchvision"] and final["libraries"]["torchaudio"]
    store = c.app.state.services.store
    seq, artifacts = store.max_seq(rid), store.artifacts(rid)
    for node in g.nodes:
        result = c.post(f"/api/runs/{rid}/inspect", json={"kind": "summary", "node": node.id}).json()
        assert result["available"] and result["summaryKind"]
        assert result["provenance"]["runId"] == rid and result["provenance"]["graphHash"] == final["graphHash"]
        assert result["provenance"]["nodeId"] == node.id and result["provenance"]["summarySha256"]
        assert result["data"]["contract"]
        if node.type == "domain.nlp_tokenizer":
            batch = result["data"]["batch"]
            assert batch["examples"] and len(batch["examples"]) == len(batch["lengths"])
            for e, length in zip(batch["examples"], batch["lengths"]):
                assert len(e["ids"]) == len(e["attention"]) == len(e["labelIds"]) == batch["shape"][1]
                assert sum(e["attention"]) == length
                assert all(y == -100 for y, mask in zip(e["labelIds"], e["attention"]) if not mask)
            tokens = next(a for a in store.artifacts(rid, "node_output") if a["meta"]["node"] == node.id)
            from tokenizers import Tokenizer
            fitted = json.loads(store.read_artifact(tokens["sha256"]))
            restored = Tokenizer.from_str(fitted["tokenizerJson"])
            assert restored.encode(batch["examples"][0]["text"]).ids == batch["examples"][0]["ids"][:batch["lengths"][0]]
    result = c.post(f"/api/runs/{rid}/inspect", json={"kind": "summary", "node": last}).json()
    assert result["data"][expected] and len(result["data"]["curve"]) == 2 and result["data"]["samples"]
    source_path = EXAMPLES.parent / g.nodes[0].config["path"]
    assert result["provenance"]["sourceSha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    # Summary content is exactly the CAS bytes, not a re-execution of any model.
    assert json.loads(store.read_artifact(result["provenance"]["summarySha256"])) == result["data"]
    assert store.max_seq(rid) == seq and store.artifacts(rid) == artifacts


def test_domain_examples_are_synthetic_and_listed_in_picker(client, domain_fixtures):
    c = client
    details = {p["id"]: p for p in c.get("/api/examples").json()["details"]}
    for name in ["vision_segmentation_synthetic", "nlp_token_classification", "speech_ctc_tones"]:
        assert details[name]["graphKind"] == "domain" and details[name]["synthetic"] is True
        ex = c.get(f"/api/examples/{name}").json()
        assert c.post("/api/validate", json={"graph": ex["graph"]}).json()["ok"]
