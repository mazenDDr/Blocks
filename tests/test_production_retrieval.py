"""Retrieval serving from pinned index snapshots (ADR 0060), offline: deterministic local_hash embeddings, model-free graph."""
import shutil

import pytest

from agent import samples as sm
from agent_helpers import Lab
from production import retrieval_agent_adapter as adapter
from production.models import PredictRequest, RegisterVersion, ReleaseCreate
from production.pipeline import ProductionError
from production.runtime import ProductionRuntime
from test_production_agent import META

DOCS = {"apples.txt": "SYNTHETIC note. Apples are red and grow on trees.",
        "bananas.txt": "SYNTHETIC note. Bananas are yellow and curved.",
        "grapes.txt": "SYNTHETIC note. Grapes are purple and small."}


def graph(docs_dir, k=1, scope="turn"):
    return sm.make_graph(
        [sm.N("find", "agent.retrieve", index="notes", query="{question}", k=k, output_field="docs"),
         sm.N("answer", "agent.set_state", assignments=[{"field": "answer", "kind": "template", "template": "SYNTHETIC answer for: {question}"}])],
        sm.chain("START", "find", "answer", "END"),
        {"state": [sm.S("question"), sm.S("docs", "documents", scope=scope), sm.S("answer")],
         "indexes": [{"id": "notes", "loader": {"directory": str(docs_dir), "glob": "*.txt"},
                      "splitter": {"strategy": "paragraph", "chunkSize": 200, "chunkOverlap": 0},
                      "embeddings": {"provider": "local_hash", "dimension": 256, "normalize": True}}],
         "limits": {"maxSteps": 8, "maxSeconds": 10, "maxModelCalls": 1, "maxTokens": 1024}})


@pytest.fixture()
def served(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in DOCS.items():
        (docs / name).write_text(text)
    lab = Lab(tmp_path / "lab")
    state, run_id = lab.run(graph(docs), inp={"question": "yellow curved bananas"})
    assert state == "completed"
    rt = ProductionRuntime(lab.store)
    version = rt.register_version(RegisterVersion(runId=run_id, node=adapter.NODE, **META))
    release = rt.create_release(ReleaseCreate(versionId=version["id"], config={"namespace": "rag", "maxBatch": 1, "captureInputs": True}))
    rt.activate(release["id"], None)
    return lab, rt, version, docs, run_id


def ask(rt, rid, text):
    t = rt.predict("local", "rag", PredictRequest(requestId=rid, records=[{"question": text}]))
    assert t["status"] == 200, t["error"]
    return t


def test_registers_a_snapshot_and_serves_retrieval_from_it(served):
    lab, rt, version, docs, run_id = served
    assert version["adapter"] == "agent_retrieval"
    snap = version["manifest"]["indexes"]["notes"]
    assert set(snap["files"]) == {"faiss.index", "chunks.json", "manifest.json"} and snap["chunks"] == 3
    assert snap["embeddingProvider"] == {"provider": "local_hash", "dimension": 256, "normalize": True}
    t = ask(rt, "q1", "yellow curved bananas")
    retrieval = t["result"]["agent"]["retrievals"][0]
    assert t["result"]["predictions"] == ["SYNTHETIC answer for: yellow curved bananas"]
    assert retrieval["index"] == "notes" and len(retrieval["included"]) == 1
    assert "bananas" in retrieval["included"][0]["chunk_id"]
    assert [e["data"]["action"] for e in t["result"]["agent"]["events"] if e["type"] == "index_ready"] == ["pinned"]


def test_changed_documents_and_deleted_live_index_do_not_change_serving(served):
    lab, rt, version, docs, run_id = served
    before = ask(rt, "q1", "yellow curved bananas")["result"]["agent"]["retrievals"]
    (docs / "bananas.txt").write_text("SYNTHETIC note. Bananas were rewritten after registration.")
    (docs / "kiwis.txt").write_text("SYNTHETIC note. Kiwis are yellow and curved inside.")
    shutil.rmtree(lab.wb / "agent" / "indexes" / "notes")
    after = ask(rt, "q2", "yellow curved bananas")["result"]["agent"]["retrievals"]
    assert after == before  # same chunk ids and scores: the pinned snapshot, not the changed folder
    assert not (lab.wb / "agent" / "indexes" / "notes").exists()  # serving never rebuilds the research index
    with pytest.raises(ProductionError) as e:  # a new registration must match what the run actually used
        rt.register_version(RegisterVersion(runId=run_id, node=adapter.NODE, **META))
    assert e.value.code == "E_AGENT_INDEX"


def test_contract_refusals(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("SYNTHETIC")
    with pytest.raises(ProductionError) as e:
        adapter.contract(graph(docs, scope="thread"), {"question": "x"})
    assert e.value.code == "E_AGENT_SCOPE"
    with pytest.raises(ProductionError) as e:
        adapter.contract(graph(docs, k=9), {"question": "x"})
    assert e.value.code == "E_AGENT_LIMITS"
    plain = graph(docs).to_json()
    plain["nodes"] = [n for n in plain["nodes"] if n["id"] != "find"]
    plain["edges"] = sm.chain("START", "answer", "END")
    plain["agent"]["state"] = [s for s in plain["agent"]["state"] if s["name"] != "docs"]
    with pytest.raises(ProductionError) as e:
        adapter.contract(sm.make_graph(plain["nodes"], plain["edges"], plain["agent"]), {"question": "x"})
    assert e.value.code == "E_AGENT_SCOPE"
