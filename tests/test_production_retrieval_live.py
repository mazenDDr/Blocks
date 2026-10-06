"""Real installed Ollama: retrieval serving with nomic embeddings and a chat model over a pinned index snapshot (ADR 0060)."""
import shutil

import pytest
from fastapi.testclient import TestClient

from agent import samples as sm
from agent.models import ollama_status
from agent_helpers import Lab
from control.app import create_app
from production import retrieval_agent_adapter as adapter
from production.models import RegisterVersion, ReleaseCreate
from production.runtime import ProductionRuntime
from test_production_agent import META

pytestmark = pytest.mark.live
DOCS = {"vault.txt": "SYNTHETIC fact sheet. The vault code word is TANGERINE-41.",
        "garden.txt": "SYNTHETIC fact sheet. The garden has three oak trees and a pond.",
        "harbor.txt": "SYNTHETIC fact sheet. The harbor ferry leaves every ninety minutes."}


def graph(docs_dir):
    return sm.make_graph(
        [sm.N("find", "agent.retrieve", index="facts", query="{question}", k=1, output_field="docs"),
         sm.N("prompt", "agent.prompt", output_field="messages", items=[
             {"kind": "template", "role": "system", "template": "Answer with the exact code word from the context only."},
             {"kind": "documents", "role": "system", "field": "docs", "header": "Context:", "template": "[{chunk_id}] {text}"},
             {"kind": "template", "role": "user", "template": "{question}"}]),
         sm.N("reply", "agent.chat_model", messages_field="messages", output_field="answer",
              model={"provider": "ollama", "model": "qwen3.5:2b", "think": False, "max_tokens": 32, "timeout_s": 20, "temperature": 0, "seed": 0})],
        sm.chain("START", "find", "prompt", "reply", "END"),
        {"state": [sm.S("question"), sm.S("docs", "documents"), sm.S("messages", "messages"), sm.S("answer")],
         "indexes": [{"id": "facts", "loader": {"directory": str(docs_dir), "glob": "*.txt"},
                      "splitter": {"strategy": "paragraph", "chunkSize": 200, "chunkOverlap": 0},
                      "embeddings": {"provider": "ollama", "model": "nomic-embed-text", "normalize": True}}],
         "limits": {"maxSteps": 8, "maxSeconds": 30, "maxModelCalls": 1, "maxTokens": 2048}})


def test_real_embeddings_and_model_answer_from_the_pinned_snapshot(tmp_path):
    status = ollama_status()
    if not status["reachable"] or not {"qwen3.5:2b", "nomic-embed-text:latest"} <= set(status["models"]):
        pytest.skip("Real installed local Ollama qwen3.5:2b and nomic-embed-text required; no downloads")
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in DOCS.items():
        (docs / name).write_text(text)
    lab = Lab(tmp_path / "lab")
    state, run_id = lab.run(graph(docs), inp={"question": "What is the vault code word?"})
    assert state == "completed"
    rt = ProductionRuntime(lab.store)
    version = rt.register_version(RegisterVersion(runId=run_id, node=adapter.NODE, **META))
    snap = version["manifest"]["indexes"]["facts"]
    assert snap["embeddingProvider"]["model"] == "nomic-embed-text:latest" and snap["embeddingProvider"]["digest"]
    release = rt.create_release(ReleaseCreate(versionId=version["id"], config={"namespace": "facts", "maxBatch": 1, "timeoutSeconds": 30, "captureInputs": True}))
    rt.activate(release["id"], None)
    # The live folder changes completely after registration; serving must keep answering from the pinned snapshot.
    (docs / "vault.txt").write_text("SYNTHETIC fact sheet. The vault code word changed to PLUM-99.")
    shutil.rmtree(lab.wb / "agent" / "indexes" / "facts")
    with TestClient(create_app(lab.wb)) as c:
        assert any(x["adapter"] == "agent_retrieval" for x in c.get("/api/production").json()["candidates"])
        r = c.post("/api/serve/local/facts/predict", json={"requestId": "v1", "records": [{"question": "What is the vault code word?"}]})
        assert r.status_code == 200, r.text
        agent = r.json()["result"]["agent"]
        assert agent["retrievals"][0]["included"][0]["chunk_id"].startswith("vault.txt")
        assert agent["retrievals"][0]["embedding"].startswith("ollama:nomic-embed-text")
        context = agent["contexts"][0]["value"]
        sent = " ".join(m["content"] for m in context["providerRequest"])
        assert "TANGERINE-41" in sent and "PLUM-99" not in sent  # the model saw the pinned chunk, not the changed file
        assert "TANGERINE" in r.json()["result"]["predictions"][0].upper()
        assert context["usage"]["source"] == "provider" and not context["fixture"]
    assert not (lab.wb / "agent" / "indexes" / "facts").exists()
