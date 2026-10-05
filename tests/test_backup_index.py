"""Native FAISS/memory recovery; lexical and live semantic evidence are separate."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agent.index import IndexStore
from agent.memory import MemoryStore
from agent.models import ollama_status
from agent.spec import IndexSpec
from artifact_store import ArtifactStore
from workbench_backup.core import create, restore


def recover_index_and_memory(tmp_path, provider):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "lab.txt").write_text("SYNTHETIC teaching protocol: the blue tube stays at four degrees Celsius.\n")
    (corpus / "travel.txt").write_text("SYNTHETIC teaching note: a red bicycle travels along the street.\n")
    source = tmp_path / "source"
    ArtifactStore(source)
    spec = IndexSpec(id="recovery", loader={"directory": str(corpus)},
                     embeddings={"provider": provider, "model": "nomic-embed-text", "normalize": True})
    index = IndexStore(source)
    built = index.ensure(spec)
    assert built["action"] == "built" and built["embeddingsComputed"] > 0
    query = "SYNTHETIC question: how cold should the blue tube remain?"
    before = index.search(spec, query, 2, None)
    assert before["documents"] and before["indexIdentity"] == built["identity"]
    memory = MemoryStore(source)
    rec = memory.put_record({"id": "fixture", "text": "SYNTHETIC version one", "namespace": "fixture", "scope": "user:fixture",
                             "source": {"kind": "labelled synthetic teaching record"}})
    memory.put_record({**rec, "text": "SYNTHETIC version two"})
    memory.delete_record("fixture", evidence={"fixture": "SYNTHETIC tombstone test"})
    args = dict(run_id="fixture", thread_id="fixture", node="fixture", tool="synthetic-ledger", args_hash="fixture")
    assert memory.claim_effect("finished", **args)[0] == "claimed"
    memory.complete_effect("finished", {"fixture": "SYNTHETIC completion; no external action"})
    assert memory.claim_effect("uncertain", **args)[0] == "claimed"
    rows, writes, effects = memory.list_records(include_deleted=True), memory.writes(), memory.effects()
    made = create(source, tmp_path / "backup", offline=True)
    shutil.rmtree(source)
    restore(tmp_path / "backup", tmp_path / "recovered", trusted=True, manifest_sha256=made["manifestSha256"])
    recovered = IndexStore(tmp_path / "recovered")
    reused = recovered.ensure(spec)
    assert reused == {**built, "action": "reused"}
    assert recovered.search(spec, query, 2, None) == before
    mem = MemoryStore(tmp_path / "recovered")
    assert mem.list_records(include_deleted=True) == rows
    assert mem.writes() == writes and len(writes) == 3
    assert mem.effects() == effects
    assert mem.claim_effect("finished", **args) == ("done", {"fixture": "SYNTHETIC completion; no external action"})
    assert mem.claim_effect("uncertain", **args) == ("uncertain", None)
    shutil.rmtree(corpus)  # External source is deliberately not included in backup.
    assert recovered.search(spec, query, 2, None) == before
    with pytest.raises(FileNotFoundError):
        recovered.ensure(spec)


def isolated_recovery(tmp_path, provider):
    # Native FAISS and torch's OpenMP runtimes abort when initialized together
    # in this Mac process order. Use a fresh process, as the native agent worker
    # does, rather than bypassing runtime checks or changing global test order.
    code = """import importlib.util,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('native_index_recovery',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.recover_index_and_memory(Path(sys.argv[2]),sys.argv[3])
print('native index/memory recovery passed')"""
    result = subprocess.run([sys.executable, "-c", code, str(Path(__file__).resolve()), str(tmp_path), provider],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "native index/memory recovery passed" in result.stdout


def test_native_faiss_lexical_index_memory_and_ledger_after_source_loss(tmp_path):
    isolated_recovery(tmp_path, "local_hash")  # Real lexical hashing; no semantic-quality claim.


@pytest.mark.live
def test_live_ollama_faiss_recovery_reuses_index_and_preserves_scores(tmp_path):
    status = ollama_status()
    if not status["reachable"] or "nomic-embed-text:latest" not in status["models"]:
        pytest.skip("Local Ollama nomic-embed-text is required; no substitute embeddings.")
    isolated_recovery(tmp_path, "ollama")
