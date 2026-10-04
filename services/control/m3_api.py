"""Milestone 3 routes: module library, module test/validate, training procedure, debugger (recorded steps, wires, gradients, capture-and-rerun, sandbox),
probes, attention inspector, code blocks.

Nothing here executes user source in the control process: code blocks run in a sandbox subprocess (codeblocks/sandbox.py)."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

from artifact_store import ArtifactStore
from codeblocks.sandbox import CodeBlockError, SandboxSession
from codeblocks.template import generate_template
from codeblocks.testing import check_block, run_fixture
from debugger import record
from debugger.attention import attention_instances, inspect_attention
from debugger.probes import measure_overhead, verify_invariants
from debugger.rerun import plan_rerun
from debugger.sandbox import SandboxError, branch
from graph_core import build as stdlib
from graph_core.composite import code_interface, module_harness
from graph_core.hashing import semantic_hash
from graph_core.lower import lower_graph
from graph_core.modtest import run_module
from graph_core.schema import CodeBlockDef, Graph, ModuleDef
from graph_core.validate import ExecutionBlocked, validate
from libstore import LibraryStore, VersionConflict
from training.data import make_datasets
from training.spec import DEFAULT_STAGES, DataSpec, ProcedureSpec
from worker.procedure_run import ProcedureRunConfig

from .m3_support import module_summaries, procedure_check
from .views import views


def err(status: int, message: str, diagnostics: list | None = None, code: str = "request_invalid") -> HTTPException:
    return HTTPException(status, {"code": code, "message": message, "diagnostics": diagnostics or []})


class PublishModule(BaseModel):
    module: dict[str, Any]
    note: str = ""


class ModuleValidate(BaseModel):
    graph: Graph
    moduleId: str
    version: str | None = None
    shapes: dict[str, list[int | str]] | None = None


class ModuleTest(BaseModel):
    graph: Graph
    moduleId: str
    version: str | None = None
    args: dict[str, Any] | None = None
    inputs: dict[str, Any]  # name -> nested list, or {"shape": [...], "seed": 0, "dtype": "float32"}
    grads: bool = True


class ProcedureDefaults(BaseModel):
    graph: Graph


class ProcedureCheck(BaseModel):
    procedure: dict[str, Any]


class WireReq(BaseModel):
    step: int
    node: str
    port: str | None = None
    limit: int = Field(64, ge=1, le=4096)


class GradReq(BaseModel):
    step: int
    node: str | None = None
    param: str | None = None
    limit: int = Field(64, ge=1, le=4096)


class CaptureReq(BaseModel):
    step: int = Field(ge=1)
    nodes: list[str] | None = None


class SandboxReq(BaseModel):
    step: int = Field(ge=1)
    interventions: list[dict[str, Any]]
    stepsForward: int = Field(1, ge=1, le=50)
    captureNodes: list[str] | None = None
    label: str = ""


class ProbeVerify(BaseModel):
    graph: Graph
    nodes: list[str] | None = None
    batch: int = Field(8, ge=1, le=256)
    repeats: int = Field(20, ge=3, le=200)


class AttentionReq(BaseModel):
    graph: Graph
    instance: str
    sample: int = 0
    head: int = 0
    token: int = 0
    runId: str | None = None
    checkpointStep: int | None = None
    batch: int = Field(4, ge=1, le=64)
    inputs: dict[str, Any] | None = None


class CodeBlockReq(BaseModel):
    block: dict[str, Any]


class CodeRunReq(BaseModel):
    block: dict[str, Any]
    fixture: dict[str, Any]


class DepsReq(BaseModel):
    pins: list[str]


def tensor_from(spec: Any, dtype: str = "float32") -> torch.Tensor:
    dt = {"float32": torch.float32, "float64": torch.float64, "int64": torch.int64, "bool": torch.bool}
    if isinstance(spec, dict) and "shape" in spec:
        g = torch.Generator().manual_seed(int(spec.get("seed", 0)))
        d = spec.get("dtype", dtype)
        if d == "int64":
            return torch.randint(int(spec.get("low", 0)), int(spec.get("high", 5)), spec["shape"], generator=g)
        if d == "bool":
            return torch.rand(spec["shape"], generator=g) > 0.5
        return torch.randn(spec["shape"], generator=g).to(dt[d])
    return torch.tensor(spec, dtype=dt[dtype])


def default_inputs(graph: Graph, batch: int) -> dict[str, torch.Tensor]:
    """Synthetic inputs of the declared types, for probe verification and attention inspection when no recorded data is given."""
    vocab = max([int(n.config.get("num_embeddings", 2)) for n in graph.nodes if n.type == "tensor.embedding"] or [2])
    out = {}
    g = torch.Generator().manual_seed(0)
    for n in graph.nodes:
        if n.type != "core.tensor_input":
            continue
        shape = [batch if d == "N" else d for d in n.config.get("shape", ["N", 4])]
        dt = n.config.get("dtype", "float32")
        if dt == "int64":
            x = torch.randint(1, max(2, vocab), shape, generator=g)
            if len(shape) == 2 and shape[1] > 2:
                x[0, shape[1] * 3 // 4:] = 0  # one padded sequence, so a padding mask has something to show
            out[n.id] = x
        elif dt == "bool":
            out[n.id] = torch.rand(shape, generator=g) > 0.3
        else:
            out[n.id] = torch.randn(shape, generator=g)
    return out


def register(app: FastAPI, sv: Any) -> None:
    store: ArtifactStore = sv.store
    modules = LibraryStore(sv.workbench, "modules")
    blocks = LibraryStore(sv.workbench, "codeblocks")
    cache_dir = sv.workbench / "codeblocks" / "cache"

    # ------------------------------------------------------------------ modules
    @app.get("/api/modules")
    def list_modules():
        return {"modules": modules.list()}

    @app.get("/api/modules/starters")
    def starters():
        """Ready-made modules built only from primitives (the same ones the examples use). Opening one shows its graph; none is code."""
        defs = [stdlib.residual_block(8), stdlib.masked_weighted_loss("valid_count"), stdlib.masked_weighted_loss("element_count"), stdlib.masked_weighted_loss("weight_sum"),
                stdlib.multi_head_attention(16, 2), stdlib.transformer_encoder_block(16, 2, 32)]
        return {"modules": [d.model_dump(mode="json", by_alias=True) for d in defs]}

    @app.get("/api/modules/{mid}")
    def get_module(mid: str, version: str | None = None):
        rec = modules.get(mid, version)
        if rec is None:
            raise HTTPException(404, {"code": "not_found", "message": f"no module '{mid}'" + (f" version {version}" if version else "")})
        return rec | {"versions": modules.versions(mid)}

    @app.post("/api/modules/publish", status_code=201)
    def publish_module(req: PublishModule):
        try:
            d = ModuleDef.model_validate(req.module)
        except ValidationError as e:
            raise err(422, "invalid module definition: " + str(e.errors()[0]["msg"]))
        g = Graph(modules=[d])
        try:
            h, _ = module_harness(g, d.id, d.version)
            r = validate(h)
        except KeyError:
            raise err(422, "module not found")
        if any(x.code in ("E_MODULE_RECURSION", "E_CYCLE", "E_BAD_ID", "E_CONFIG") and x.severity == "error" for x in r.diagnostics):
            raise err(422, "the module has errors and cannot be published", [x.to_json() for x in r.errors])
        try:
            rec = modules.publish(d.model_dump(mode="json", by_alias=True), req.note)
        except VersionConflict as e:
            raise HTTPException(409, {"code": "version_immutable", "message": str(e), "nextVersion": LibraryStore.next_version(d.version)})
        except ValueError as e:
            raise err(422, str(e))
        return rec

    @app.post("/api/modules/validate")
    def validate_module(req: ModuleValidate):
        """Validate one module on its own: it is instantiated on tensor inputs (declared shapes, unknown dims 4). Inner node ids come back without the harness prefix."""
        mod = req.graph.module(req.moduleId, req.version)
        if mod is None:
            raise HTTPException(404, {"code": "not_found", "message": f"no module '{req.moduleId}' in this graph"})
        h, inst = module_harness(req.graph, req.moduleId, req.version, req.shapes)
        r = validate(h)
        v = views(h, r)
        strip = lambda k: k[len(inst) + 1:] if k.startswith(inst + "/") else k  # noqa: E731
        flat = {strip(k): val for k, val in {**v["flat"]}.items()}
        diags = [d.to_json() | {"nodeId": strip(d.nodeId) if d.nodeId else None} for d in r.diagnostics]
        info = r.expansion.instances.get(inst)
        outs = {p: r.output_types[ep[0]][ep[1]].to_json() for p, ep in (info.out_map.items() if info else []) if ep[0] in r.output_types}
        return {"ok": r.ok, "diagnostics": diags, "nodes": flat, "outputs": outs, "params": sum(r.params.get(m, 0) for m in (info.members if info else [])),
                "moduleHash": semantic_hash(h), "inputShapes": {n.id[3:]: n.config["shape"] for n in h.nodes if n.id.startswith("in_")}}

    @app.post("/api/modules/test")
    def test_module(req: ModuleTest):
        mod = req.graph.module(req.moduleId, req.version)
        if mod is None:
            raise HTTPException(404, {"code": "not_found", "message": f"no module '{req.moduleId}' in this graph"})
        ports = {p.name: p for p in mod.inputs}
        try:
            ins = {k: tensor_from(v, ports[k].dtype if k in ports else "float32") for k, v in req.inputs.items()}
            res = run_module(req.graph, req.moduleId, ins, req.version, req.args, req.grads)
        except ExecutionBlocked as e:
            raise err(422, "the module has errors and cannot run", [d.to_json() for d in e.diagnostics], "execution_blocked")
        except (KeyError, RuntimeError, ValueError) as e:
            raise err(422, f"{type(e).__name__}: {e}")
        enc = lambda t: {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""), "values": t.detach().double().flatten().tolist()[:512]}  # noqa: E731
        return {"outputs": {k: enc(v) for k, v in res["outputs"].items()}, "grads": {k: enc(v) for k, v in res.get("grads", {}).items() if v is not None},
                "reduction": mod.reduction, "provenance": {"kind": "computed now from the inputs given; not a recorded run", "moduleHash": semantic_hash(req.graph)}}

    # ------------------------------------------------------------------ training procedure
    @app.post("/api/procedure/check")
    def check_proc(req: ProcedureCheck):
        return procedure_check(req.procedure)

    @app.post("/api/procedure/default")
    def default_proc(req: ProcedureDefaults):
        g = req.graph
        inputs = [n for n in g.nodes if n.type == "core.tensor_input"]
        spec = ProcedureSpec()
        if len(inputs) == 1:
            shape, dt = inputs[0].config.get("shape", ["N", 4]), inputs[0].config.get("dtype", "float32")
            vocab = max([int(n.config.get("num_embeddings", 12)) for n in g.nodes if n.type == "tensor.embedding"] or [12])
            if dt == "int64" and len(shape) == 2:
                spec.data = DataSpec(kind="synthetic_sequence", seq_len=shape[1], vocab=max(3, vocab))
            elif len(shape) == 2:
                spec.data = DataSpec(kind="synthetic_regression", features=shape[1])
                spec.loss.kind = "mse"
                spec.loss.ports = {"pred": "output", "target": "y"}
        return {"procedure": spec.model_dump(mode="json"), "stages": DEFAULT_STAGES, "note": "Defaults for this graph; the data is SYNTHETIC and generated from data_seed."}

    @app.get("/api/runs/{rid}/procedure")
    def run_procedure_view(rid: str):
        row = store.get_run(rid)
        if row is None or row["config"].get("kind") != "procedure":
            raise HTTPException(404, {"code": "not_found", "message": f"no training-procedure run '{rid}'"})
        evs = store.events(rid, -1, ("train_step", "validation", "scheduler_step", "early_stopping_check", "checkpoint", "checkpoint_pruned", "breakpoint_hit", "assertion_failed", "optimizer_trace", "epoch_end", "capture_recorded", "rerun_verified"))
        by = lambda t: [{"seq": e["seq"], **e["data"], **({"node": e["node_id"]} if e["node_id"] else {})} for e in evs if e["type"] == t]  # noqa: E731
        steps = by("train_step")
        started = store.last_event(rid, "run_started")
        return {"runId": rid, "procedure": row["config"]["procedure"], "stages": row["config"]["procedure"].get("stages", DEFAULT_STAGES), "status": row["status"],
                "run": started["data"] if started else None,
                "steps": [{k: s.get(k) for k in ("step", "epoch", "loss", "lr", "grad_norm", "window", "clip")} for s in steps],
                "validations": by("validation"), "epochs": by("epoch_end"), "schedulerSteps": by("scheduler_step"), "earlyStopping": by("early_stopping_check"),
                "checkpoints": [{"step": c["step"], "status": c["status"], **c["meta"], "sha256": c["sha256"]} for c in store.artifacts(rid, "checkpoint")],
                "pruned": by("checkpoint_pruned"), "breakpoints": by("breakpoint_hit") + by("assertion_failed"), "optimizerTrace": by("optimizer_trace"),
                "rerunVerification": by("rerun_verified")}

    @app.get("/api/runs/{rid}/optimizer")
    def optimizer_view(rid: str, step: int | None = None, param: str | None = None):
        """Optimizer state stored in a checkpoint (VISION 10.2): the real moments / momentum buffers and step counts, per parameter."""
        row = store.get_run(rid)
        if row is None or row["config"].get("kind") != "procedure":
            raise HTTPException(404, {"code": "not_found", "message": f"no training-procedure run '{rid}'"})
        cks = [c for c in store.artifacts(rid, "checkpoint") if c["status"] != "pruned" and c["meta"].get("tag") != "best"]
        if step is not None:
            cks = [c for c in cks if c["step"] == step]
        if not cks:
            return {"available": False, "reason": "no_checkpoint", "message": "no checkpoint at that step: optimizer state is stored only in checkpoints (and for the watched element in optimizer_trace events)."}
        c = max(cks, key=lambda c: c["id"])
        st = torch.load(__import__("io").BytesIO(store.read_artifact(c["sha256"])), weights_only=True)
        spec = ProcedureSpec.model_validate(row["config"]["procedure"])
        opt = st["optimizer"]
        # parameter order in the optimizer = order of trainable parameters = order of model.parameters() among those requiring grad
        names = st.get("param_names", [])
        out = []
        for i, pstate in opt["state"].items():
            entry: dict[str, Any] = {"index": int(i), "name": names[int(i)] if int(i) < len(names) else None}
            for k, v in pstate.items():
                if torch.is_tensor(v):
                    t = v.double()
                    entry[k] = {"shape": list(v.shape), "mean": float(t.mean()), "norm": float(t.norm()), "min": float(t.min()), "max": float(t.max())} if v.numel() > 1 else {"value": float(t)}
            out.append(entry)
        if param:
            out = [o for o in out if o["name"] == param]
        return {"available": True, "runId": rid, "step": c["step"], "optimizer": spec.optimizer.model_dump(), "paramGroups": [{k: v for k, v in g.items() if k != "params"} for g in opt["param_groups"]],
                "state": out, "scheduler": st.get("scheduler"), "rngCaptured": True, "provenance": {"runId": rid, "checkpoint": c["sha256"], "step": c["step"], "kind": "stored optimizer state"}}

    # ------------------------------------------------------------------ debugger
    def procedure_run(rid: str):
        row = store.get_run(rid)
        if row is None:
            raise HTTPException(404, {"code": "not_found", "message": f"unknown run '{rid}'"})
        if row["config"].get("kind") != "procedure":
            raise err(422, f"run '{rid}' is not a training-procedure run (the debugger records those)")
        return row

    @app.get("/api/runs/{rid}/debug/steps")
    def debug_steps(rid: str):
        procedure_run(rid)
        return record.scrubber(store, rid)

    @app.post("/api/runs/{rid}/debug/wire")
    def debug_wire(rid: str, req: WireReq):
        procedure_run(rid)
        return record.wire_value(store, rid, req.step, req.node, req.port, req.limit)

    @app.post("/api/runs/{rid}/debug/gradients")
    def debug_grads(rid: str, req: GradReq):
        procedure_run(rid)
        return record.gradient_view(store, rid, req.step, req.node, req.param, req.limit)

    @app.post("/api/runs/{rid}/debug/capture", status_code=201)
    def debug_capture(rid: str, req: CaptureReq):
        """Capture and rerun: a NEW run that re-executes the original deterministically (from its nearest checkpoint) with a capture at the step. The original is not touched."""
        procedure_run(rid)
        cfg = plan_rerun(store, rid, req.step, req.nodes)
        gs = store.artifacts(rid, "graph")
        graph = Graph.model_validate(json.loads(store.read_artifact(gs[0]["sha256"])))
        from worker.process import submit_run

        with sv.lock:
            handle = submit_run(graph, cfg, sv.workbench)
            sv.handles[handle.run_id] = handle
        return {"runId": handle.run_id, "rerunOf": rid, "step": req.step, "resumeFrom": cfg.resume_from, "status": "queued"}

    @app.post("/api/runs/{rid}/debug/sandbox", status_code=201)
    def debug_sandbox(rid: str, req: SandboxReq):
        procedure_run(rid)
        try:
            return branch(store, rid, req.step, req.interventions, req.stepsForward, req.captureNodes, req.label)
        except SandboxError as e:
            raise err(422, str(e), code="sandbox_invalid")
        except ExecutionBlocked as e:
            raise err(422, str(e), [d.to_json() for d in e.diagnostics], "execution_blocked")

    @app.get("/api/runs/{rid}/debug/sandboxes")
    def debug_sandboxes(rid: str):
        procedure_run(rid)
        out = []
        for r in store.list_runs():
            if r["config"].get("kind") == "sandbox" and r["config"].get("parent") == rid:
                ev = store.last_event(r["id"], "sandbox_result")
                if ev:
                    d = ev["data"]
                    out.append({"sandboxRunId": r["id"], "step": r["config"]["step"], "label": r["config"].get("label"), "changed": d["changed"], "outcome": d["outcome"],
                                "lossDelta": d["diff"]["lossAtStep"]["delta"], "createdAt": r["created_at"]})
        return {"sandboxes": out}

    @app.get("/api/runs/{rid}/debug/sandboxes/{sid}")
    def debug_sandbox_get(rid: str, sid: str):
        ev = store.last_event(sid, "sandbox_result") if (store.get_run(sid) or {}).get("config", {}).get("parent") == rid else None
        if ev is None:
            raise HTTPException(404, {"code": "not_found", "message": f"no sandbox '{sid}' for run '{rid}'"})
        return ev["data"]

    @app.post("/api/debug/probes/verify")
    def probes_verify(req: ProbeVerify):
        """Verify that probes on `nodes` change nothing (outputs, gradients, parameters, RNG, mode, order) and measure their overhead on this model."""
        try:
            model = lower_graph(req.graph)
        except ExecutionBlocked as e:
            raise err(422, "the graph has errors and cannot run", [d.to_json() for d in e.diagnostics], "execution_blocked")
        ins = default_inputs(req.graph, req.batch)
        args = [ins[i] for i in model.input_ids]
        bad = [n for n in (req.nodes or []) if n not in model._modules]
        if bad:
            raise err(422, f"unknown nodes {bad}")
        inv = verify_invariants(model, args, req.nodes)
        ov = measure_overhead(model, args, req.nodes, repeats=req.repeats)
        return {"invariants": inv, "overhead": ov, "mode": "observation only (forward hooks on detached copies)", "batch": req.batch,
                "inputs": "synthetic inputs of the declared types (not recorded data)"}

    # ------------------------------------------------------------------ attention
    @app.post("/api/attention/instances")
    def attn_instances(req: ProcedureDefaults):
        return {"instances": attention_instances(req.graph)}

    @app.post("/api/attention/inspect")
    def attn_inspect(req: AttentionReq):
        g = req.graph
        state, note = None, "initial weights (seeded initialisation; no run selected, so the model has not been trained)"
        seed = 0
        data_note = "SYNTHETIC batch of the declared input types (generated, not recorded data)"
        inputs: dict[str, torch.Tensor] = {}
        if req.inputs:
            inputs = {k: tensor_from(v, "int64" if isinstance(v, list) and v and isinstance(v[0], list) and isinstance(v[0][0], int) else "float32") for k, v in req.inputs.items()}
            data_note = "inputs given in the request"
        if req.runId:
            row = procedure_run(req.runId)
            spec = ProcedureSpec.model_validate(row["config"]["procedure"])
            cks = [c for c in store.artifacts(req.runId, "checkpoint") if c["status"] != "pruned" and c["meta"].get("tag") != "best"]
            if req.checkpointStep is not None:
                cks = [c for c in cks if c["step"] == req.checkpointStep]
            if cks:
                c = max(cks, key=lambda c: (c["step"], c["id"]))
                state = torch.load(__import__("io").BytesIO(store.read_artifact(c["sha256"])), weights_only=True)["model"]
                note = f"weights of run {req.runId} at optimizer step {c['step']} (checkpoint {c['sha256'][:10]})"
            else:
                note = f"run {req.runId} has no checkpoint{'' if req.checkpointStep is None else ' at step ' + str(req.checkpointStep)}: initial weights shown"
            if not inputs:
                _, val = make_datasets(spec.data)
                model = lower_graph(g)
                field = {nid: spec.data.inputs.get(nid, "x") for nid in model.input_ids}
                inputs = {nid: val.fields[f][:req.batch] for nid, f in field.items()}
                data_note = f"first {req.batch} validation examples of run {req.runId} ({val.description})"
        elif not inputs:
            inputs = default_inputs(g, req.batch)
        try:
            out = inspect_attention(g, req.instance, inputs, sample=req.sample, head=req.head, token=req.token, state=state, weights_note=note, seed=seed)
        except ExecutionBlocked as e:
            raise err(422, "the graph has errors and cannot run", [d.to_json() for d in e.diagnostics], "execution_blocked")
        except RuntimeError as e:
            raise err(422, f"could not run the model on these inputs: {e}")
        out["dataNote"] = data_note
        return out

    # ------------------------------------------------------------------ code blocks
    def parse_block(raw: dict[str, Any]) -> CodeBlockDef:
        try:
            return CodeBlockDef.model_validate(raw)
        except ValidationError as e:
            raise err(422, "invalid code block definition: " + "; ".join(f"{'.'.join(str(x) for x in x_['loc'])}: {x_['msg']}" for x_ in e.errors()))

    @app.post("/api/codeblocks/template")
    def cb_template(req: CodeBlockReq):
        return {"source": generate_template(parse_block(req.block))}

    @app.post("/api/codeblocks/test")
    def cb_test(req: CodeBlockReq):
        d = parse_block(req.block)
        if not d.fixtures:
            raise err(422, "add at least one fixture (inputs with shapes, optionally expected outputs) to test the block")
        return check_block(d, cache_dir)

    @app.post("/api/codeblocks/run")
    def cb_run(req: CodeRunReq):
        d = parse_block(req.block)
        return run_fixture(d, req.fixture)

    @app.get("/api/codeblocks")
    def cb_list():
        return {"codeBlocks": blocks.list()}

    @app.get("/api/codeblocks/environment")
    def cb_env():
        """The environment the blocks run in: the control service's own interpreter, as seen from inside a sandbox process."""
        s = SandboxSession({"id": "env", "version": "0", "effects": [], "randomness": "none", "differentiable": False, "dependencies": [], "limits": {"wall_seconds": 60}},
                           "def run():\n    pass\n")
        try:
            e = s.environment()
        except CodeBlockError as ex:
            raise err(500, ex.message)
        finally:
            s.close()
        return {"python": e["python"], "torch": e["torch"], "packages": e["packages"],
                "note": "Dependencies are pinned by name==version and checked when a block loads; installing packages is not implemented in this version."}

    @app.post("/api/codeblocks/dependencies")
    def cb_deps(req: DepsReq):
        s = SandboxSession({"id": "env", "version": "0", "effects": [], "randomness": "none", "differentiable": False, "dependencies": [], "limits": {"wall_seconds": 60}},
                           "def run():\n    pass\n")
        try:
            pk = {p.split("==")[0].lower().replace("_", "-"): p.split("==")[1] for p in s.environment()["packages"]}
        finally:
            s.close()
        out = []
        for pin in req.pins:
            name, _, want = pin.partition("==")
            have = pk.get(name.strip().lower().replace("_", "-"))
            out.append({"pin": pin, "installed": have, "status": "missing" if have is None else ("ok" if (not want or have == want.strip()) else "version_mismatch")})
        return {"pins": out}

    @app.post("/api/codeblocks/publish", status_code=201)
    def cb_publish(req: CodeBlockReq):
        d = parse_block(req.block)
        try:
            return blocks.publish(d.model_dump(mode="json"), "") | {"identity": code_interface(d)["identity"]}
        except VersionConflict as e:
            raise HTTPException(409, {"code": "version_immutable", "message": str(e), "nextVersion": LibraryStore.next_version(d.version)})
        except ValueError as e:
            raise err(422, str(e))

    @app.get("/api/codeblocks/{bid}")
    def cb_get(bid: str, version: str | None = None):
        rec = blocks.get(bid, version)
        if rec is None:
            raise HTTPException(404, {"code": "not_found", "message": f"no code block '{bid}'"})
        return rec | {"versions": blocks.versions(bid)}
