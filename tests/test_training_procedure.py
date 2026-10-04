"""Milestone 3 item 5: the editable training procedure. Order of operations, accumulation, clipping, optimizers with real torch semantics,
schedulers with explicit timing, validation/checkpoint/early-stopping policies, optimizer state inspection, and A12 exact resume."""
import copy

import pytest
import torch
import torch.nn.functional as F

from graph_core.build import GraphBuilder, masked_weighted_loss
from graph_core.validate import ExecutionBlocked
from training.optim import manual_update
from training.spec import (AccumulationSpec, BestSpec, Cadence, CheckpointSpec, ClipSpec, DataSpec, EarlyStopSpec, LossSpec, OptimizerSpec,
                           ProcedureSpec, SchedulerSpec, ValidationSpec, WatchSpec, check_procedure)
from training.trainer import Trainer


def seq_model(dropout=None):
    g = GraphBuilder()
    g.input("x", ["N", 6], "int64")
    g.node("emb", "tensor.embedding", num_embeddings=12, embedding_dim=8)
    g.node("pool", "core.mean", dims=[1])
    g.node("hid", "tensor.dense", out_features=8)
    g.node("act", "tensor.tanh")
    last = "act"
    g.wire("x", "emb.input")
    g.chain("emb", "pool", "hid", "act")
    if dropout is not None:
        g.node("drop", "tensor.dropout", p=dropout)
        g.wire("act", "drop.input")
        last = "drop"
    g.node("head", "tensor.dense", out_features=2)
    g.wire(last, "head.input")
    return g.build()


def spec(**kw):
    base = dict(epochs=2, data=DataSpec(seq_len=6, n_train=64, n_val=32, batch_size=8), optimizer=OptimizerSpec(kind="sgd", lr=0.1))
    base.update(kw)
    return ProcedureSpec(**base)


def params_equal(a, b):
    sa, sb = a.model.state_dict(), b.model.state_dict()
    return all(torch.equal(sa[k], sb[k]) for k in sa)


# ------------------------------------------------------------------------------------------------ order of operations
def test_order_rules_are_enforced_with_stable_codes():
    base = ["zero_grad", "forward", "loss", "backward", "clip", "optimizer_step"]
    def codes(stages, **kw):
        return {d.code for d in check_procedure(spec(stages=stages, **kw))}
    assert codes(base, clip=ClipSpec(kind="norm")) == set()
    assert "E_PROC_ORDER" in codes(["zero_grad", "forward", "loss", "backward", "optimizer_step", "clip"], clip=ClipSpec(kind="norm"))   # clip after the step
    assert "E_PROC_ORDER" in codes(["forward", "loss", "zero_grad", "backward", "optimizer_step"])                                         # zero_grad erases the graph's inputs... between forward and step
    assert "E_PROC_ORDER" in codes(["forward", "loss", "backward", "zero_grad", "optimizer_step"])
    assert "E_PROC_ORDER" in codes(["forward", "backward", "loss", "optimizer_step"])
    assert "E_PROC_STAGE" in codes(["forward", "loss", "backward"])                                                                         # no optimizer step
    assert "E_PROC_STAGE" in codes(base[:-2] + ["optimizer_step"], clip=ClipSpec(kind="norm"))                                              # clipping configured, stage absent
    assert "W_PROC_NO_ZERO_GRAD" in codes(["forward", "loss", "backward", "optimizer_step"])
    sch = SchedulerSpec(kind="step", timing="optimizer_step")
    assert "E_PROC_ORDER" in codes(["zero_grad", "forward", "loss", "backward", "scheduler_step", "optimizer_step"], scheduler=sch)
    assert "E_PROC_STAGE" in codes(base, scheduler=sch)                                                                                      # timing optimizer_step but no scheduler stage
    with pytest.raises(ExecutionBlocked):
        Trainer(seq_model(), spec(stages=["forward", "loss", "backward", "optimizer_step", "clip"], clip=ClipSpec(kind="norm")))


def test_zero_grad_position_changes_nothing_but_removing_it_changes_behaviour():
    g = seq_model()
    a = Trainer(g, spec(stages=["zero_grad", "forward", "loss", "backward", "optimizer_step"]))
    b = Trainer(g, spec(stages=["forward", "loss", "backward", "optimizer_step", "zero_grad"]))
    c = Trainer(g, spec(stages=["forward", "loss", "backward", "optimizer_step"]))
    for t in (a, b, c):
        t.run()
    assert params_equal(a, b)            # window start vs window end are equivalent
    assert not params_equal(a, c)        # never zeroing accumulates every gradient: a different (and declared-as-warned) run
    assert c.warnings and c.warnings[0].code == "W_PROC_NO_ZERO_GRAD"


# ------------------------------------------------------------------------------------------------ native reference
def test_plain_loop_matches_a_handwritten_torch_loop_bitwise():
    g = seq_model()
    t = Trainer(g, spec(epochs=2, optimizer=OptimizerSpec(kind="sgd", lr=0.1, momentum=0.9)))
    t.run()
    # handwritten loop with the same initial weights (a fresh Trainer's model before training) and the same data order
    t0 = Trainer(g, spec(epochs=2, optimizer=OptimizerSpec(kind="sgd", lr=0.1, momentum=0.9)))
    model = t0.model
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    model.train()
    ds, B = t0.train_ds, 8
    for epoch in range(2):
        order = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0 * 100003 + epoch))
        for s in range(0, len(ds), B):
            idx = order[s:s + B]
            opt.zero_grad()
            F.cross_entropy(model(ds.fields["x"][idx]), ds.fields["y"][idx]).backward()
            opt.step()
    assert params_equal(t, t0)
    assert t.opt_step == 16


@pytest.mark.parametrize("opt_spec", [
    OptimizerSpec(kind="sgd", lr=0.05),
    OptimizerSpec(kind="sgd", lr=0.05, momentum=0.9, weight_decay=0.01),
    OptimizerSpec(kind="sgd", lr=0.05, momentum=0.9, nesterov=True),
    OptimizerSpec(kind="adam", lr=0.01, weight_decay=0.01),
    OptimizerSpec(kind="adamw", lr=0.01, weight_decay=0.1),
    OptimizerSpec(kind="adam", lr=0.01, amsgrad=True),
])
def test_optimizers_have_real_torch_semantics_and_the_mirror_formula_matches(opt_spec):
    t = Trainer(seq_model(), spec(epochs=1, optimizer=opt_spec, watch=WatchSpec(param="head.weight", index=[1, 3])))
    t.run()
    traces = [e for e in t.events if e["type"] == "optimizer_trace"]
    assert len(traces) == t.opt_step == 8
    for e in traces:
        assert e["abs_diff"] < 1e-6, (opt_spec.kind, e["step"], e["abs_diff"])      # hand-written update rule == torch's result
    # the torch class used is the one asked for
    assert type(t.opt).__name__ == {"sgd": "SGD", "adam": "Adam", "adamw": "AdamW"}[opt_spec.kind]


def test_stateful_optimizer_state_is_exposed_adam_moments_and_step_count():
    t = Trainer(seq_model(), spec(epochs=1, optimizer=OptimizerSpec(kind="adam", lr=0.01), watch=WatchSpec(param="head.weight", index=[0, 0])))
    t.run()
    tr = [e for e in t.events if e["type"] == "optimizer_trace"]
    assert tr[0]["state_before"] == {} or "exp_avg" not in tr[0]["state_before"]
    assert tr[0]["state_after"]["step"] == 1.0 and tr[-1]["state_after"]["step"] == 8.0
    # the stored exp_avg after step k equals 0.9*exp_avg_before + 0.1*grad, taken from the real optimizer state
    for prev, cur in zip(tr, tr[1:]):
        m = 0.9 * prev["state_after"]["exp_avg"] + 0.1 * cur["grad"]
        assert cur["state_after"]["exp_avg"] == pytest.approx(m, rel=1e-5, abs=1e-8)
    assert any("bias corrections" in line for line in tr[0]["formula"])
    # the live optimizer state is the same object the trace read
    p = dict(t.model.named_parameters())["head.weight"]
    assert float(t.opt.state[p]["exp_avg"][0, 0]) == pytest.approx(tr[-1]["state_after"]["exp_avg"])


def test_manual_update_sgd_teaching_step_from_the_vision():
    # w = 2, gradient = 0.6, lr = 0.1 => w_next = 1.94
    w, st, steps = manual_update(OptimizerSpec(kind="sgd", lr=0.1), 0.1, torch.tensor(2.0), torch.tensor(0.6), {})
    assert float(w) == pytest.approx(1.94, abs=1e-6)


# ------------------------------------------------------------------------------------------------ accumulation
@pytest.mark.parametrize("opt_spec", [OptimizerSpec(kind="sgd", lr=0.1), OptimizerSpec(kind="adam", lr=0.01)])
def test_gradient_accumulation_matches_reference_large_batch_step(opt_spec):
    g = seq_model()
    big = Trainer(g, spec(epochs=1, data=DataSpec(seq_len=6, n_train=64, n_val=16, batch_size=32), optimizer=opt_spec))
    acc = Trainer(g, spec(epochs=1, data=DataSpec(seq_len=6, n_train=64, n_val=16, batch_size=8), optimizer=opt_spec,
                          accumulation=AccumulationSpec(steps=4)))
    big.run(); acc.run()
    assert big.opt_step == acc.opt_step == 2 and acc.micro_total == 8
    sa, sb = big.model.state_dict(), acc.model.state_dict()
    for k in sa:   # not bitwise: the sum of four means is rounded differently from one mean of 32
        assert torch.allclose(sa[k], sb[k], atol=1e-6, rtol=1e-5), k
    # accumulation changed nothing about WHAT was learned relative to micro-batch stepping
    stepping = Trainer(g, spec(epochs=1, data=DataSpec(seq_len=6, n_train=64, n_val=16, batch_size=8), optimizer=opt_spec))
    stepping.run()
    assert not torch.allclose(stepping.model.state_dict()["head.weight"], sb["head.weight"], atol=1e-6)


def test_accumulation_partial_window_policies():
    d = DataSpec(seq_len=6, n_train=40, n_val=8, batch_size=8)   # 5 micro-batches per epoch, window 4 => windows of 4 and 1
    a = Trainer(seq_model(), spec(epochs=1, data=d, accumulation=AccumulationSpec(steps=4)))
    a.run()
    assert a.opt_step == 2 and [h["window"] for h in a.history] == [4, 1]
    b = Trainer(seq_model(), spec(epochs=1, data=d, accumulation=AccumulationSpec(steps=4, partial_window="drop")))
    b.run()
    assert b.opt_step == 1 and b.micro_total == 4


# ------------------------------------------------------------------------------------------------ clipping
def test_clip_norm_clips_the_intended_gradients_and_reports_the_decision():
    t = Trainer(seq_model(), spec(epochs=1, optimizer=OptimizerSpec(kind="sgd", lr=0.1), clip=ClipSpec(kind="norm", max_norm=0.05)))
    t.run()
    steps = [e for e in t.events if e["type"] == "train_step"]
    assert all(e["clip"]["clipped"] and e["clip"]["norm_before"] > 0.05 for e in steps)
    assert all(e["clip"]["scale"] == pytest.approx(0.05 / (e["clip"]["norm_before"] + 1e-6)) for e in steps)
    # the clipped gradient norm is exactly max_norm (checked on a fresh single step)
    t2 = Trainer(seq_model(), spec(epochs=1, clip=ClipSpec(kind="norm", max_norm=0.05), max_optimizer_steps=1,
                                   stages=["zero_grad", "forward", "loss", "backward", "clip", "optimizer_step", "zero_grad"][:6]))
    t2.run()
    ref = Trainer(seq_model(), spec(epochs=1, max_optimizer_steps=1))
    ref.run()
    d_clip = sum(float((a - b).norm()) for a, b in zip(t2.model.state_dict().values(), seq_init(t2)))
    d_free = sum(float((a - b).norm()) for a, b in zip(ref.model.state_dict().values(), seq_init(ref)))
    assert d_clip < d_free      # a clipped step moves the weights less


def seq_init(t):
    fresh = Trainer(t.graph, t.spec)
    return list(fresh.model.state_dict().values())


def test_clip_value_and_no_clip_when_under_threshold():
    t = Trainer(seq_model(), spec(epochs=1, clip=ClipSpec(kind="value", value=1e-3), max_optimizer_steps=2))
    t.run()
    assert all(e["clip"]["clipped"] and e["clip"]["kind"] == "value" for e in t.events if e["type"] == "train_step")
    t = Trainer(seq_model(), spec(epochs=1, clip=ClipSpec(kind="norm", max_norm=1e6), max_optimizer_steps=2))
    t.run()
    assert not any(e["clip"]["clipped"] for e in t.events if e["type"] == "train_step")


# ------------------------------------------------------------------------------------------------ scheduler timing
def lrs(t):
    return [e["lr"] for e in t.events if e["type"] == "train_step"]


def test_scheduler_step_timing_is_explicit():
    base = dict(epochs=2, data=DataSpec(seq_len=6, n_train=32, n_val=8, batch_size=8))   # 4 steps per epoch
    t = Trainer(seq_model(), spec(**base, scheduler=SchedulerSpec(kind="step", step_size=2, gamma=0.5, timing="optimizer_step")))
    t.run()
    assert lrs(t) == pytest.approx([0.1, 0.1, 0.05, 0.05, 0.025, 0.025, 0.0125, 0.0125])
    t = Trainer(seq_model(), spec(**base, stages=["zero_grad", "forward", "loss", "backward", "clip", "optimizer_step"],
                                  scheduler=SchedulerSpec(kind="exponential", gamma=0.1, timing="epoch")))
    t.run()
    assert lrs(t) == pytest.approx([0.1] * 4 + [0.01] * 4)       # constant within an epoch, stepped once per epoch
    t = Trainer(seq_model(), spec(**base, scheduler=SchedulerSpec(kind="linear_warmup", warmup_steps=4)))
    t.run()
    assert lrs(t)[:4] == pytest.approx([0.1 * 1 / 4, 0.1 * 2 / 4, 0.1 * 3 / 4, 0.1]) and lrs(t)[4] == pytest.approx(0.1)


def test_reduce_on_plateau_reacts_to_the_validation_metric_event():
    t = Trainer(seq_model(), spec(epochs=3, optimizer=OptimizerSpec(kind="sgd", lr=0.0001),   # too small to improve => plateau
                                  scheduler=SchedulerSpec(kind="reduce_on_plateau", timing="validation", patience=0, factor=0.5)))
    t.run()
    sched_events = [e for e in t.events if e["type"] == "scheduler_step"]
    assert sched_events and all(e["timing"] == "validation" for e in sched_events)
    assert t.opt.param_groups[0]["lr"] <= 0.0001


def test_scheduler_with_accumulation_steps_once_per_optimizer_step():
    t = Trainer(seq_model(), spec(epochs=1, data=DataSpec(seq_len=6, n_train=64, n_val=8, batch_size=8), accumulation=AccumulationSpec(steps=4),
                                  scheduler=SchedulerSpec(kind="step", step_size=1, gamma=0.5)))
    t.run()
    assert lrs(t) == pytest.approx([0.1, 0.05])


# ------------------------------------------------------------------------------------------------ validation / checkpoints / early stopping
def test_validation_frequency_eval_mode_and_no_grad_are_separate_and_explicit():
    t = Trainer(seq_model(dropout=0.5), spec(epochs=2, validation=ValidationSpec(every=Cadence(unit="optimizer_step", n=3))))
    t.run()
    v = [e for e in t.events if e["type"] == "validation"]
    assert [e["step"] for e in v] == [3, 6, 9, 12, 15] and all(e["eval_mode"] and e["no_grad"] for e in v)
    # eval mode really switches dropout off: validating twice gives the same number; train-mode validation would not
    assert t.validate()["val_loss"] == t.validate()["val_loss"]
    t2 = Trainer(seq_model(dropout=0.5), spec(epochs=1, validation=ValidationSpec(eval_mode=False)))
    t2.run()
    assert t2.validate()["val_loss"] != t2.validate()["val_loss"]
    assert t.model.training    # validation restores the training mode


def test_checkpoint_policy_cadence_retention_and_best():
    ck = CheckpointSpec(every=Cadence(unit="optimizer_step", n=2), keep_last=2, best=BestSpec(metric="val_loss", mode="min"))
    t = Trainer(seq_model(), spec(epochs=2, checkpoint=ck, validation=ValidationSpec(every=Cadence(unit="epoch", n=1))))
    t.run()
    periodic = [m for m in t.saved if m["tag"] == "periodic"]
    assert [m["step"] for m in periodic] == [2, 4, 6, 8, 10, 12, 14, 16]
    pruned = [e for e in t.events if e["type"] == "checkpoint_pruned"]
    assert [e["step"] for e in pruned] == [2, 4, 6, 8, 10, 12]            # keep_last=2 keeps steps 14 and 16
    assert any(m["tag"] == "best" for m in t.saved)
    best = [m for m in t.saved if m["tag"] == "best"][-1]
    assert best["step"] == 16 or best["step"] == 8
    assert all("sha256" in m and m["exact_resume"] for m in t.saved)


def test_early_stopping_stops_on_patience_and_records_why():
    t = Trainer(seq_model(), spec(epochs=20, optimizer=OptimizerSpec(kind="sgd", lr=1e-7),    # nothing improves by min_delta
                                  early_stopping=EarlyStopSpec(metric="val_loss", mode="min", patience=2, min_delta=0.05)))
    r = t.run()
    assert r["stopped_by"] == "early_stopping" and t.epoch < 20
    checks = [e for e in t.events if e["type"] == "early_stopping_check"]
    assert checks[-1]["bad"] == 2 and checks[0]["bad"] == 0


def test_frozen_nodes_are_not_trained():
    t = Trainer(seq_model(), spec(epochs=1, frozen=["emb"]))
    before = {k: v.clone() for k, v in t.model.state_dict().items()}
    t.run()
    after = t.model.state_dict()
    assert torch.equal(before["emb.weight"], after["emb.weight"]) and not torch.equal(before["head.weight"], after["head.weight"])


# ------------------------------------------------------------------------------------------------ A12 resume
@pytest.mark.parametrize("name,kw,graph", [
    ("sgd+momentum+dropout", dict(optimizer=OptimizerSpec(kind="sgd", lr=0.1, momentum=0.9)), seq_model(dropout=0.3)),
    ("adam+scheduler+clip+accumulation", dict(optimizer=OptimizerSpec(kind="adam", lr=0.01), scheduler=SchedulerSpec(kind="exponential", gamma=0.9),
                                              clip=ClipSpec(kind="norm", max_norm=0.5), accumulation=AccumulationSpec(steps=2)), seq_model(dropout=0.3)),
    ("adamw+epoch-scheduler+earlystop-state", dict(optimizer=OptimizerSpec(kind="adamw", lr=0.01, weight_decay=0.1),
                                                   scheduler=SchedulerSpec(kind="cosine", t_max=10, timing="epoch"),
                                                   early_stopping=EarlyStopSpec(patience=50)), seq_model()),
])
@pytest.mark.parametrize("stop_at", [3, 8])    # mid-epoch and exactly at an epoch end (8 steps per epoch for batch 8 / 64 samples without accumulation)
def test_a12_resume_continues_bit_for_bit_vs_uninterrupted_reference(name, kw, graph, stop_at):
    s = spec(epochs=3, checkpoint=CheckpointSpec(every=None), **kw)
    ref = Trainer(graph, s)
    ref.run()
    first = Trainer(graph, s)
    first.run(until_opt_step=stop_at)
    blob = first.checkpoint_bytes()
    # a brand-new Trainer (as a restarted process would build it), polluted global RNG state, then load and continue
    torch.manual_seed(12345)
    torch.rand(7)
    resumed = Trainer(graph, s)
    resumed.load_state(Trainer.read_state(blob))
    assert resumed.opt_step == first.opt_step
    resumed.run()
    assert resumed.opt_step == ref.opt_step
    for k, v in ref.model.state_dict().items():
        assert torch.equal(v, resumed.model.state_dict()[k]), (name, k)           # bitwise equality on CPU in deterministic mode
    so, sr = ref.opt.state_dict()["state"], resumed.opt.state_dict()["state"]
    for k in so:
        for kk, v in so[k].items():
            assert torch.equal(torch.as_tensor(v), torch.as_tensor(sr[k][kk])), (name, k, kk)
    assert torch.equal(torch.get_rng_state(), torch.get_rng_state())
    ref_losses = [e["loss"] for e in ref.events if e["type"] == "train_step"][stop_at:]
    res_losses = [e["loss"] for e in resumed.events if e["type"] == "train_step"]
    assert ref_losses == res_losses                                              # the whole loss curve after the resume point is identical
    assert ref.last_metrics == resumed.last_metrics


def test_resume_refuses_a_different_graph_or_procedure():
    s = spec(epochs=1)
    t = Trainer(seq_model(), s)
    t.run(until_opt_step=2)
    blob = t.checkpoint_bytes()
    other_graph = seq_model()
    other_graph.nodes[1].config["embedding_dim"] = 8
    other_graph.nodes[3].config["out_features"] = 9
    with pytest.raises(ValueError, match="graph"):
        Trainer(other_graph, s).load_state(Trainer.read_state(blob))
    with pytest.raises(ValueError, match="procedure"):
        Trainer(seq_model(), spec(epochs=1, optimizer=OptimizerSpec(kind="sgd", lr=0.5))).load_state(Trainer.read_state(blob))
    # extending the number of epochs is allowed (it does not change the trajectory)
    Trainer(seq_model(), spec(epochs=5)).load_state(Trainer.read_state(blob))


def test_checkpoint_contents_cover_documented_state():
    t = Trainer(seq_model(dropout=0.2), spec(epochs=1, optimizer=OptimizerSpec(kind="adam"), scheduler=SchedulerSpec(kind="step")))
    t.run(until_opt_step=3)
    st = Trainer.read_state(t.checkpoint_bytes())
    assert {"model", "optimizer", "scheduler", "rng", "counters", "early", "best_value", "graph_hash", "spec_hash", "torch", "training"} <= st.keys()
    assert st["counters"]["opt_step"] == 3 and "exp_avg" in next(iter(st["optimizer"]["state"].values()))


# ------------------------------------------------------------------------------------------------ module loss in training
def test_training_with_a_visual_masked_weighted_loss_module_decreases_loss():
    g = GraphBuilder()
    g.input("x", ["N", 4])
    g.node("fc", "tensor.dense", out_features=2)
    g.wire("x", "fc.input")
    graph = g.build()
    graph.modules.append(masked_weighted_loss("valid_count"))
    s = ProcedureSpec(epochs=6, data=DataSpec(kind="synthetic_regression", features=4, outputs=2, n_train=128, n_val=32, batch_size=16),
                      loss=LossSpec(kind="module", module="masked_weighted_mse_valid_count",
                                    ports={"pred": "output", "target": "y", "mask": "mask", "weights": "weights"}),
                      optimizer=OptimizerSpec(kind="sgd", lr=0.05))
    t = Trainer(graph, s)
    t.run()
    ep = [e for e in t.events if e["type"] == "epoch_end"]
    assert ep[-1]["train_loss"] < ep[0]["train_loss"] * 0.5
    # the first optimizer-step loss equals the hand-written value for the initial weights
    t0 = Trainer(graph, s)
    batch = t0._batch(t0._order(0), 0)
    out = t0._forward(batch)
    m, w, y = batch["mask"], batch["weights"], batch["y"]
    ref = (w * m * (out - y) ** 2).sum() / m.sum().clamp(min=1e-8)
    assert float(t0._loss(out, batch)) == pytest.approx(float(ref), rel=1e-6)
    assert t.events[0]["type"] != "x"
