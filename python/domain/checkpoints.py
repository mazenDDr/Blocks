"""Internal native domain checkpoints; no whole-module pickle or uploaded code."""
from __future__ import annotations
import hashlib
import io
import json
import platform
from importlib.metadata import version
from pathlib import Path
import torch
from tabular.core import ExecutionError, dumps

FAMILIES = {"vision", "nlp", "speech"}
MAX_CHECKPOINT = 32 * 1024 * 1024


def environment():
    return {"python": platform.python_version(), "threads": torch.get_num_threads(),
            **{n: version(n) for n in ("torch", "torchvision", "torchaudio", "tokenizers", "numpy")}}


def implementation():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / "domain/inference.py"]
    for folder in ("vision", "nlp", "speech", "domain"):
        paths.extend((root / folder).glob("*.py"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def fingerprint(config, contract, tensors=()):
    """Bind resolved training settings and actual prepared data, not just source names."""
    h = hashlib.sha256(dumps({"config": config, "contract": contract}).encode())
    for t in tensors:
        a = t.detach().cpu().contiguous().numpy()
        h.update(str(a.dtype).encode()); h.update(dumps(list(a.shape)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def training_state(model, opt, generator, epochs, curve, **extra):
    return {"weights": model.state_dict(), "optimizer": opt.state_dict(), "generator": generator.get_state(),
            "torchRng": torch.get_rng_state(), "epochs": epochs, "curve": curve, **extra}


def restore(model, opt, generator, saved, epochs):
    if saved is None:
        return 0, []
    if epochs <= saved["epochs"]:
        raise ExecutionError("E_DOMAIN_RESUME_EPOCHS", "Total epochs must exceed the recorded completed epoch count.")
    model.load_state_dict(saved["weights"], strict=True)
    opt.load_state_dict(saved["optimizer"])
    generator.set_state(saved["generator"])
    torch.set_rng_state(saved["torchRng"])
    return saved["epochs"], list(saved["curve"])


def verified(store, sha):
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha) or not store.verify(sha):
        raise ExecutionError("E_DOMAIN_CHECKPOINT_INTEGRITY", "Checkpoint artifact is missing or fails its SHA-256 check.")
    raw = store.read_artifact(sha)
    if len(raw) > MAX_CHECKPOINT:
        raise ExecutionError("E_DOMAIN_CHECKPOINT_BOUNDS", "Checkpoint exceeds the 32 MiB limit.")
    return raw


def read_manifest(store, identity):
    raw = verified(store, identity)
    matches = [a for r in store.list_runs() for a in store.artifacts(r["id"], "domain_model")
               if a["sha256"] == identity and a["status"] == "complete" and a["meta"].get("internalDomainModel")]
    if not matches:
        raise ExecutionError("E_DOMAIN_CHECKPOINT_TRUST", "Only domain models recorded by this workbench can be loaded.")
    m = json.loads(raw)
    if m.get("format") != "void-domain-model/1" or m.get("family") not in FAMILIES:
        raise ExecutionError("E_DOMAIN_CHECKPOINT_FORMAT", "Unsupported checkpoint format or model family.")
    if not any(a["run_id"] == m["runId"] and a["meta"].get("node") == m["node"] for a in matches):
        raise ExecutionError("E_DOMAIN_CHECKPOINT_TRUST", "Model manifest does not belong to its declared run/node.")
    row = store.get_run(m["runId"])
    if not row or row["status"] != "completed" or row["graph_hash"] != m["graphHash"]:
        raise ExecutionError("E_DOMAIN_CHECKPOINT_RUN", "Inference and resume require a completed matching source run.")
    # Thread count is relevant to exact resume; inference does not promise bitwise CPU scheduling.
    current = environment()
    if {k:v for k,v in m["environment"].items() if k != "threads"} != {k:v for k,v in current.items() if k != "threads"} or m["implementation"] != implementation():
        raise ExecutionError("E_DOMAIN_CHECKPOINT_ENVIRONMENT", "Pinned native environment or implementation differs.")
    if not any(a["kind"] == "domain_checkpoint" and a["sha256"] == m["checkpointSha256"] and a["meta"].get("internalDomainModel")
               and a["meta"].get("node") == m["node"] for a in store.artifacts(m["runId"])):
        raise ExecutionError("E_DOMAIN_CHECKPOINT_TRUST", "Checkpoint is not a recorded internal artifact of this model.")
    return m


def read_state(store, manifest):
    try:
        return torch.load(io.BytesIO(verified(store, manifest["checkpointSha256"])), map_location="cpu", weights_only=True)
    except ExecutionError:
        raise
    except Exception as e:
        raise ExecutionError("E_DOMAIN_CHECKPOINT_FORMAT", f"Native state dictionary cannot be loaded: {type(e).__name__}") from e


def resume(ctx, identity, family, signature):
    if not identity:
        return None
    if ctx.store is None:
        raise ExecutionError("E_DOMAIN_CHECKPOINT_STORE", "Resume requires the owning workbench artifact store.")
    m = read_manifest(ctx.store, identity)
    if m["family"] != family or m["trainingSignature"] != signature or m["environment"]["threads"] != torch.get_num_threads():
        raise ExecutionError("E_DOMAIN_RESUME_INCOMPATIBLE", "Prepared data, split, tokenizer, training configuration or CPU thread count differs.")
    return read_state(ctx.store, m)


def persist(ctx, family, signature, state, inference, source, parent=None, example=None):
    if ctx.store is None or ctx.run_id is None:
        return None  # Pure native numerical calls do not invent a run/artifact identity.
    if any(not bool(torch.isfinite(t).all()) for t in state["weights"].values()):
        raise ExecutionError("E_DOMAIN_CHECKPOINT_NONFINITE", "Learned parameters are nonfinite; checkpoint refused.")
    buf = io.BytesIO(); torch.save(state, buf)
    if buf.tell() > MAX_CHECKPOINT:
        raise ExecutionError("E_DOMAIN_CHECKPOINT_BOUNDS", "Checkpoint exceeds the 32 MiB limit.")
    meta = {"node": ctx.node_id, "internalDomainModel": True, "family": family}
    a = ctx.store.add_artifact(ctx.run_id, "domain_checkpoint", buf.getvalue(), "complete", state["epochs"], meta)
    example_ref = None
    if example is not None:
        example_ref = ctx.store.add_artifact(ctx.run_id, "domain_inference_example", dumps(example).encode(), "complete", state["epochs"], meta)["sha256"]
    m = {"format": "void-domain-model/1", "family": family, "runId": ctx.run_id, "graphHash": ctx.graph_hash, "node": ctx.node_id,
         "checkpointSha256": a["sha256"], "epochs": state["epochs"], "trainingSignature": signature,
         "environment": environment(), "implementation": implementation(), "inference": inference, "source": source,
         "exampleSha256": example_ref, "parentModelId": parent, "resumeBoundary": "completed epoch; identical prepared CPU data and configuration only"}
    a = ctx.store.add_artifact(ctx.run_id, "domain_model", dumps(m).encode(), "complete", state["epochs"], meta)
    return {"modelId": a["sha256"], "checkpointSha256": m["checkpointSha256"], "epochs": state["epochs"], "parentModelId": parent}
