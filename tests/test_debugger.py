"""Milestone 3 item 7 / 8: the debugger. Probes that observe only (A35, overhead measured), conditional breakpoints labelled as execution-changing,
recorded steps with honest 'not recorded' (A29), wires with real values (A28), gradients, capture-and-rerun verified against the original, sandbox
interventions that leave the original immutable and record what changed (A36), and nested diagnostic blocks (A46)."""
import copy
import json

import pytest
import torch

from artifact_store import ArtifactStore
from debugger import record
from debugger.capture import StepCapture
from debugger.probes import ProbeSet, measure_overhead, verify_invariants
from debugger.rerun import plan_rerun
from debugger.sandbox import SandboxError, branch, fingerprint
from graph_core import diagnostics as D
from graph_core.build import GraphBuilder, ModuleBuilder
from graph_core.hashing import semantic_hash
from graph_core.lower import lower_graph
from graph_core.validate import validate
from training.spec import (AccumulationSpec, Breakpoint, CaptureSpec, ClipSpec, DataSpec, LossSpec, OptimizerSpec, ProbeSpec, ProcedureSpec, SchedulerSpec, WatchSpec)
from training.trainer import Trainer
from worker.procedure_run import ProcedureRunConfig, run_procedure
from test_training_procedure import seq_model


def make_run(tmp_path, graph, run_id="r1", cfg=None, store=None, **spec_kw):
    store = store or ArtifactStore(tmp_path)
    base = dict(epochs=2, data=DataSpec(seq_len=6, n_train=64, n_val=16, batch_size=8), optimizer=OptimizerSpec(kind="sgd", lr=0.1))
    base.update(spec_kw)
    spec = ProcedureSpec(**base)
    cfg = cfg or ProcedureRunConfig(procedure=spec.model_dump(mode="json"))
    store.create_run(run_id, semantic_hash(graph), cfg.model_dump())
    store.add_artifact(run_id, "graph", json.dumps(graph.to_json(), sort_keys=True).encode(), "complete", None, {})
    status = run_procedure(graph, cfg, store, run_id)
    return store, spec, status


# ------------------------------------------------------------------------------------------------ the procedure as a recorded run
def test_procedure_run_records_events_checkpoints_and_exposes_optimizer_state(tmp_path):
    store, spec, status = make_run(tmp_path, seq_model(), optimizer=OptimizerSpec(kind="adam", lr=0.01), watch=WatchSpec(param="head.weight", index=[0, 1]),
                                   checkpoint={"every": {"unit": "optimizer_step", "n": 4}, "keep_last": 2})
    assert status == "completed"
    types = {e["type"] for e in store.events("r1")}
    assert {"run_started", "train_step", "validation", "epoch_end", "optimizer_trace", "checkpoint", "checkpoint_pruned", "run_finished"} <= types
    started = store.last_event("r1", "run_started")["data"]
    assert started["synthetic"] is True and "SYNTHETIC" in started["data"] and started["deterministic"] is True
    cks = store.artifacts("r1", "checkpoint")
    assert [c["status"] for c in cks if c["step"] == 4] == ["pruned"] and [c["status"] for c in cks if c["step"] == 16] == ["complete"]
    st = Trainer.read_state(store.read_artifact(cks[-1]["sha256"]))
    assert st["counters"]["opt_step"] == 16 and st["optimizer"]["state"][0]["step"].item() == 16.0     # Adam's real step count is in the stored state


def test_resume_from_a_stored_checkpoint_in_a_second_run_matches_the_uninterrupted_run(tmp_path):
    g = seq_model(dropout=0.2)
    store, spec, _ = make_run(tmp_path, g, "ref", checkpoint={"every": {"unit": "optimizer_step", "n": 5}}, optimizer=OptimizerSpec(kind="adam", lr=0.01))
    cfg_ref = ProcedureRunConfig.model_validate(store.get_run("ref")["config"])
    cfg2 = cfg_ref.model_copy(update={"resume_from": {"run_id": "ref", "step": 5}})
    store.create_run("res", semantic_hash(g), cfg2.model_dump())
    store.add_artifact("res", "graph", json.dumps(g.to_json(), sort_keys=True).encode(), "complete", None, {})
    assert run_procedure(g, cfg2, store, "res") == "completed"
    la = {e["data"]["step"]: e["data"]["loss"] for e in store.events("ref", -1, ("train_step",))}
    lb = {e["data"]["step"]: e["data"]["loss"] for e in store.events("res", -1, ("train_step",))}
    assert sorted(lb) == list(range(6, 17)) and all(la[s] == lb[s] for s in lb)     # bit-for-bit equal losses
    fa, fb = (Trainer.read_state(store.read_artifact(store.artifacts(r, "checkpoint")[-1]["sha256"])) for r in ("ref", "res"))
    assert all(torch.equal(fa["model"][k], fb["model"][k]) for k in fa["model"])
    assert store.last_event("res", "run_started")["data"]["resumed_from"]["step"] == 5


# ------------------------------------------------------------------------------------------------ A35 probes
def test_a35_probes_keep_outputs_gradients_params_rng_mode_and_order_and_overhead_is_measured():
    g = seq_model(dropout=0.3)
    m = lower_graph(g)
    x = torch.randint(1, 12, (16, 6))
    inv = verify_invariants(m, [x], nodes=None, seed=3)
    assert inv["ok"], inv
    assert inv["outputs_identical"] and inv["gradients_identical"] and inv["rng_state_identical"] and inv["execution_order_identical"] and inv["parameters_and_buffers_identical"]
    assert inv["probed_nodes"] == len(m._modules)
    inv2 = verify_invariants(m, [x], nodes=["hid", "head"], seed=3)
    assert inv2["ok"] and inv2["probed_nodes"] == 2
    ov = measure_overhead(m, [x], nodes=None, repeats=15)
    assert ov["baselineMs"] > 0 and ov["probedMs"] > 0 and ov["repeats"] == 30 and "median" in ov["note"]
    print(f"\nprobe overhead (all {ov['probedNodes']} nodes, forward+backward, batch 16): {ov['baselineMs']:.3f} ms -> {ov['probedMs']:.3f} ms ({ov['overheadPercent']:+.1f}%)")


def test_probe_in_a_run_records_values_without_changing_the_trajectory(tmp_path):
    g = seq_model()
    _, _, _ = make_run(tmp_path / "a", g, "plain")
    store_b, _, _ = make_run(tmp_path / "b", g, "probed", probes=[ProbeSpec(node="hid", values=4)])
    store_a = ArtifactStore(tmp_path / "a")
    la = [e["data"]["loss"] for e in store_a.events("plain", -1, ("train_step",))]
    lb = [e["data"]["loss"] for e in store_b.events("probed", -1, ("train_step",))]
    assert la == lb                                                        # bit-identical losses with the probe on
    probes = store_b.events("probed", -1, ("probe",))
    assert len(probes) == 16                                               # one record per micro-batch forward: 8 per epoch x 2 epochs
    assert probes[0]["node_id"] == "hid" and len(probes[0]["data"]["values"]) == 4 and probes[0]["data"]["nonFinite"] == 0


# ------------------------------------------------------------------------------------------------ conditional breakpoints
def overflow_regression_model():
    """Dense -> tanh -> dense regression model. Under MSE, one SGD step at lr=1e30 leaves every weight finite (gradients are O(1) and the
    hidden activations are bounded by tanh), but the next prediction is ~1e30 and its square overflows float32 (max ~3.4e38) by ~20 orders
    of magnitude. The first non-finite loss is therefore at optimizer step 2 on any platform, with finite weights before it."""
    g = GraphBuilder()
    g.input("x", ["N", 4])
    g.node("hid", "tensor.dense", out_features=8)
    g.node("act", "tensor.tanh")
    g.node("head", "tensor.dense", out_features=2)
    g.wire("x", "hid.input")
    g.chain("hid", "act", "head")
    return g.build()


def test_conditional_breakpoint_is_labelled_execution_changing_and_stops_resumably(tmp_path):
    g = overflow_regression_model()
    store, spec, status = make_run(tmp_path, g, optimizer=OptimizerSpec(kind="sgd", lr=1e30), loss=LossSpec(kind="mse"),
                                   data=DataSpec(kind="synthetic_regression", features=4, outputs=2, n_train=64, n_val=16, batch_size=8),
                                   breakpoints=[Breakpoint(id="bad_loss", kind="nonfinite_loss")])
    assert status == "completed"
    hit = store.events("r1", -1, ("breakpoint_hit",))
    assert len(hit) == 1 and hit[0]["data"]["execution_changing"] is True and hit[0]["data"]["breakpoint"] == "bad_loss"
    assert store.events("r1", -1, ("run_finished",))[0]["data"]["stopped_by"] == "breakpoint:bad_loss"
    step = hit[0]["data"]["step"]
    assert step == 2                                                              # finite after one update, overflowing on the next forward
    ck = [c for c in store.artifacts("r1", "checkpoint") if c["meta"].get("tag") == "breakpoint"]
    assert ck and ck[-1]["step"] == step - 1                                      # the state BEFORE the offending step is what was kept
    st = Trainer.read_state(store.read_artifact(ck[-1]["sha256"]))
    assert all(torch.isfinite(v).all() for k, v in st["model"].items() if v.dtype.is_floating_point)


@pytest.mark.parametrize("bp,expect", [
    (Breakpoint(id="g", kind="grad_norm_above", threshold=1e-9), "gradient norm"),
    (Breakpoint(id="l", kind="loss_above", threshold=0.0), "exceeds"),
    (Breakpoint(id="s", kind="step_reached", step=3), "step 3 reached"),
    (Breakpoint(id="n", kind="nonfinite_node", node="hid"), None),
])
def test_other_breakpoint_kinds(tmp_path, bp, expect):
    store, _, _ = make_run(tmp_path, seq_model(), breakpoints=[bp])
    hit = store.events("r1", -1, ("breakpoint_hit",))
    if expect is None:
        assert not hit            # a healthy model never trips a non-finite breakpoint: it observes, it does not fire spuriously
    else:
        assert len(hit) == 1 and expect in hit[0]["data"]["detail"]
    if bp.kind == "step_reached":
        assert hit[0]["data"]["step"] == 3


def test_nonfinite_node_breakpoint_fires_on_the_node(tmp_path):
    g = GraphBuilder()
    g.input("x", ["N", 6], "int64")
    g.node("emb", "tensor.embedding", num_embeddings=12, embedding_dim=4)
    g.node("pool", "core.mean", dims=[1])
    g.node("scale", "core.scalar_mul", factor=1e38)
    g.node("big", "tensor.mul")
    g.node("head", "tensor.dense", out_features=2)
    g.wire("x", "emb.input"); g.chain("emb", "pool", "scale")
    g.wire("scale", "big.a"); g.wire("scale", "big.b"); g.wire("big", "head.input")
    store, _, _ = make_run(tmp_path, g.build(), breakpoints=[Breakpoint(id="nf", kind="nonfinite_node", node="big")])
    hit = store.events("r1", -1, ("breakpoint_hit",))
    assert hit and "node 'big'" in hit[0]["data"]["detail"]


# ------------------------------------------------------------------------------------------------ A28/A29 recorded steps, wires, gradients
def captured_run(tmp_path, **kw):
    return make_run(tmp_path, seq_model(), capture=CaptureSpec(steps=[2, 6], nodes=None), **kw)


def test_a29_scrubber_marks_captured_steps_and_says_not_recorded_for_the_rest(tmp_path):
    store, _, _ = captured_run(tmp_path)
    s = record.scrubber(store, "r1")
    assert [x["step"] for x in s["steps"]] == list(range(1, 17)) and s["captureSteps"] == [2, 6]
    assert [x["step"] for x in s["steps"] if x["captured"]] == [2, 6]
    nr = record.wire_value(store, "r1", 3, "hid")
    assert nr["available"] is False and nr["reason"] == "not_recorded" and "not recorded" in nr["message"]
    assert nr["action"]["kind"] == "capture_and_rerun" and nr["action"]["body"] == {"step": 3, "nodes": ["hid"]}
    assert "values" not in nr and "summary" not in nr          # no fabricated numbers
    g = record.gradient_view(store, "r1", 4)
    assert g["available"] is False and g["reason"] == "not_recorded"


def test_a28_wire_value_is_the_real_activation_of_that_step(tmp_path):
    store, spec, _ = captured_run(tmp_path)
    w = record.wire_value(store, "r1", 6, "hid", "output")
    assert w["available"] and w["summary"]["shape"] == [8, 8] and w["provenance"]["runId"] == "r1" and w["provenance"]["step"] == 6 and w["provenance"]["captured"]
    # independent recomputation: replay the run to step 5, take the first micro-batch of step 6, run the forward pass
    t = Trainer(seq_model(), spec)
    t.run(until_opt_step=5)
    order = t._order(t.epoch)
    batch = t._batch(order, t.micro_in_epoch)
    cap = StepCapture(t.model, ["hid"]).attach()
    t.model(batch["x"])
    cap.detach()
    assert [round(v, 6) for v in w["summary"]["values"][:16]] == [round(float(v), 6) for v in cap.activations["hid"].flatten()[:16]]
    assert w["provenance"]["graphHash"] == store.get_run("r1")["graph_hash"]


def test_gradients_tab_samples_captured_step_with_accumulation_scaling_labelled(tmp_path):
    store, spec, _ = make_run(tmp_path, seq_model(), capture=CaptureSpec(steps=[2], nodes=None), accumulation=AccumulationSpec(steps=2))
    ov = record.gradient_view(store, "r1", 2)
    assert ov["available"] and "head" in ov["nodes"] and "head.weight" in ov["parameters"] and "x0.5" not in ov["provenance"]["note"]
    assert "loss*0.5" in ov["provenance"]["note"] and "accumulation scaling included" in ov["provenance"]["note"]
    pg = record.gradient_view(store, "r1", 2, param="head.weight")
    assert pg["summary"]["shape"] == [2, 8] and pg["summary"]["norm"] > 0
    # check against torch: the first micro-batch's gradient of head.weight, scaled by 1/2
    t = Trainer(seq_model(), spec)
    t.run(until_opt_step=1)
    batch = t._batch(t._order(t.epoch), t.micro_in_epoch)
    t.opt.zero_grad()
    (t._loss(t._forward(batch), batch) * 0.5).backward()
    ref = dict(t.model.named_parameters())["head.weight"].grad
    assert torch.allclose(torch.tensor(pg["summary"]["values"]).reshape(2, 8), ref, atol=1e-7)
    nd = record.gradient_view(store, "r1", 2, node="emb")
    assert nd["available"] and nd["output"]["shape"] == [8, 6, 8] and "emb.weight" in nd["parameters"]
    none = record.gradient_view(store, "r1", 2, node="positions_does_not_exist")
    assert none["available"] is False


def test_capture_is_observation_only_the_run_with_capture_equals_the_run_without(tmp_path):
    a, _, _ = make_run(tmp_path / "a", seq_model(dropout=0.3), "plain")
    b, _, _ = make_run(tmp_path / "b", seq_model(dropout=0.3), "cap", capture=CaptureSpec(steps=[1, 2, 3, 4], nodes=None))
    la = [e["data"]["loss"] for e in a.events("plain", -1, ("train_step",))]
    lb = [e["data"]["loss"] for e in b.events("cap", -1, ("train_step",))]
    assert la == lb
    sa = Trainer.read_state(a.read_artifact(a.artifacts("plain", "checkpoint")[-1]["sha256"]))
    sb = Trainer.read_state(b.read_artifact(b.artifacts("cap", "checkpoint")[-1]["sha256"]))
    assert all(torch.equal(sa["model"][k], sb["model"][k]) for k in sa["model"]) and torch.equal(sa["rng"]["torch_cpu"], sb["rng"]["torch_cpu"])


def test_capture_and_rerun_fills_in_a_missing_step_and_is_verified_against_the_original(tmp_path):
    store, spec, _ = make_run(tmp_path, seq_model(dropout=0.2), checkpoint={"every": {"unit": "optimizer_step", "n": 4}}, optimizer=OptimizerSpec(kind="adam", lr=0.01))
    assert record.wire_value(store, "r1", 7, "hid")["available"] is False
    cfg = plan_rerun(store, "r1", 7, ["hid", "head"])
    assert cfg.resume_from == {"run_id": "r1", "max_step": 6} and cfg.until_step == 7 and cfg.rerun_of == "r1"
    store.create_run("rr", semantic_hash(seq_model(dropout=0.2)), cfg.model_dump())
    store.add_artifact("rr", "graph", json.dumps(seq_model(dropout=0.2).to_json(), sort_keys=True).encode(), "complete", None, {})
    assert run_procedure(seq_model(dropout=0.2), cfg, store, "rr") == "completed"
    ver = store.events("rr", -1, ("rerun_verified",))[0]["data"]
    assert ver["identical"] is True and ver["compared_steps"] == [5, 6, 7]      # resumed from the step-4 checkpoint; every recomputed loss is bit-identical to the original
    w = record.wire_value(store, "rr", 7, "hid")
    assert w["available"] and w["provenance"]["step"] == 7
    assert record.wire_value(store, "rr", 7, "emb")["available"] is False     # only the requested nodes were captured: still honest about the rest
    # the original run is unchanged: it still has no capture at step 7
    assert record.wire_value(store, "r1", 7, "hid")["available"] is False


# ------------------------------------------------------------------------------------------------ A36 sandbox
def test_a36_sandbox_branch_original_immutable_changes_recorded_and_baseline_equals_original(tmp_path):
    store, spec, _ = make_run(tmp_path, seq_model(), checkpoint={"every": {"unit": "optimizer_step", "n": 3}})
    before = fingerprint(store, "r1")
    ck_files_before = {c["sha256"]: store.verify(c["sha256"]) for c in store.artifacts("r1", "checkpoint")}
    res = branch(store, "r1", 5, [{"kind": "hyper", "path": "optimizer.lr", "value": 2.0}], steps_forward=3, label="lr x20")
    assert res["immutability"]["originalRunUnchanged"] and res["immutability"]["fingerprintBefore"] == res["immutability"]["fingerprintAfter"] == before
    assert fingerprint(store, "r1") == before and all(store.verify(s) for s in ck_files_before)
    # replay of the original state was verified and the baseline reproduces what the original run recorded
    assert res["parent"]["reconstruction"]["replayIdenticalToRecorded"] and res["parent"]["reconstruction"]["startedFrom"] == "checkpoint"
    orig = {e["data"]["step"]: e["data"]["loss"] for e in store.events("r1", -1, ("train_step",))}
    assert res["baseline"]["losses"] == [orig[5], orig[6], orig[7]]
    # the intervention is recorded: what changed, what was held fixed, and the outcome
    assert res["changed"] == [{"kind": "hyper", "target": "optimizer.lr", "before": 0.1, "after": 2.0}]
    assert any("data order" in h for h in res["heldFixed"]) and any("random state" in h for h in res["heldFixed"])
    d = res["diff"]
    assert d["lossAtStep"]["baseline"] == res["baseline"]["losses"][0] and d["lossAtStep"]["delta"] == d["lossAtStep"]["branch"] - d["lossAtStep"]["baseline"]
    assert res["branch"]["losses"][0] == res["baseline"]["losses"][0]            # lr acts at the update, so step 5's own loss is unchanged...
    assert res["branch"]["losses"][1] != res["baseline"]["losses"][1]            # ...and step 6 (after the bigger update) differs
    assert d["paramsL2DistanceBranchVsBaseline"] > 0 and "optimizer step 5" in res["outcome"]
    # stored as a new run of kind 'sandbox'
    sb = store.get_run(res["sandboxRunId"])
    assert sb["config"]["kind"] == "sandbox" and sb["config"]["parent"] == "r1" and sb["status"] == "completed"
    assert store.events(res["sandboxRunId"])[0]["data"]["changed"] == res["changed"]


def test_sandbox_activation_and_parameter_interventions_change_execution_only_inside_the_sandbox(tmp_path):
    store, spec, _ = make_run(tmp_path, seq_model())
    before = fingerprint(store, "r1")
    res = branch(store, "r1", 3, [{"kind": "activation", "node": "hid", "op": "zero", "index": None}], capture_nodes=["hid", "act", "head"])
    assert res["changed"][0]["kind"] == "activation" and res["changed"][0]["after"] == 0.0
    assert res["diff"]["activationMaxAbsDiff"]["hid"] > 0 and res["diff"]["activationMaxAbsDiff"]["head"] > 0
    assert res["diff"]["lossAtStep"]["branch"] != res["diff"]["lossAtStep"]["baseline"]
    res2 = branch(store, "r1", 3, [{"kind": "parameter", "name": "head.weight", "op": "set", "index": [0, 0], "value": 5.0}])
    assert res2["changed"][0]["after"] == 5.0 and res2["changed"][0]["before"] != 5.0 and res2["diff"]["lossAtStep"]["delta"] != 0
    res3 = branch(store, "r1", 3, [{"kind": "activation", "node": "hid", "op": "scale", "index": [0, 1], "value": 0.0}], capture_nodes=["hid"])
    assert res3["diff"]["activationMaxAbsDiff"]["hid"] > 0
    assert fingerprint(store, "r1") == before
    # an intervention that changes nothing gives an identical outcome (a control)
    ctl = branch(store, "r1", 3, [{"kind": "hyper", "path": "optimizer.lr", "value": 0.1}])
    assert ctl["diff"]["lossAtStep"]["delta"] == 0 and ctl["diff"]["paramsL2DistanceBranchVsBaseline"] == 0


def test_sandbox_rejects_unsupported_or_malformed_interventions(tmp_path):
    store, _, _ = make_run(tmp_path, seq_model())
    for bad in ([], [{"kind": "hyper", "path": "model.depth", "value": 3}], [{"kind": "activation", "node": "hid", "op": "explode"}], [{"kind": "nope"}],
                [{"kind": "parameter", "name": "no.such", "op": "zero"}], [{"kind": "activation", "node": "ghost", "op": "zero"}]):
        with pytest.raises(SandboxError):
            branch(store, "r1", 3, bad)
    with pytest.raises(SandboxError):
        branch(store, "r1", 0, [{"kind": "hyper", "path": "optimizer.lr", "value": 1}])
    # a failed sandbox leaves no trace in the original
    assert fingerprint(store, "r1")["status"] == "completed"


def test_sandbox_without_checkpoints_replays_from_the_seeded_initialisation(tmp_path):
    store, _, _ = make_run(tmp_path, seq_model(), checkpoint={"every": None})
    res = branch(store, "r1", 4, [{"kind": "hyper", "path": "clip.max_norm", "value": 0.001}])
    assert res["parent"]["reconstruction"]["startedFrom"] == "seeded initialisation" and res["parent"]["reconstruction"]["replayIdenticalToRecorded"]
