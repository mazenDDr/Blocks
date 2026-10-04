"""Write the Milestone 3 example projects (graph JSON + separate UI JSON) into examples/. Deterministic: running it twice gives identical files.

    python examples/make_m3_examples.py

Every graph is built from the same Python objects the tests use (python/graph_core/build.py); the files are ordinary project documents the editor opens."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "python")]

from graph_core.build import (GraphBuilder, ModuleBuilder, masked_weighted_loss, residual_block, transformer_classifier)  # noqa: E402
from graph_core.hashing import semantic_hash  # noqa: E402
from graph_core.schema import CodeBlockDef, CodeIO, Graph  # noqa: E402
from graph_core.validate import validate  # noqa: E402
from training.spec import (AccumulationSpec, CheckpointSpec, Cadence, ClipSpec, DataSpec, EarlyStopSpec, LossSpec, OptimizerSpec, ProcedureSpec,  # noqa: E402
                           SchedulerSpec, ValidationSpec, WatchSpec)


def layout(graph: Graph, step_x: int = 260) -> dict:
    """Left-to-right layout by topological depth (layout only; never part of the semantic hash)."""
    depth: dict[str, int] = {}
    srcs: dict[str, list[str]] = {}
    for e in graph.edges:
        srcs.setdefault(e.to.node, []).append(e.from_.node)
    def d(n: str) -> int:
        if n not in depth:
            depth[n] = 0
            depth[n] = 1 + max([d(s) for s in srcs.get(n, []) if s in {x.id for x in graph.nodes}] or [-1])
        return depth[n]
    cols: dict[int, int] = {}
    pos = {}
    for n in graph.nodes:
        c = d(n.id)
        r = cols.get(c, 0)
        cols[c] = r + 1
        pos[n.id] = {"x": 60 + c * step_x, "y": 80 + r * 150}
    return pos


def write(name: str, graph: Graph, description: str, synthetic: bool) -> None:
    r = validate(graph)
    assert r.ok, (name, [d.message for d in r.diagnostics])
    (HERE / f"{name}.project.json").write_text(json.dumps(graph.to_json(), indent=2, ensure_ascii=False) + "\n")
    ui = {"schemaVersion": "1.0.0", "positions": layout(graph), "description": description, "synthetic": synthetic}
    (HERE / f"{name}.ui.json").write_text(json.dumps(ui, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {name}: {len(graph.nodes)} top-level nodes, {r.total_params} parameters, graph {semantic_hash(graph)[:12]}")


def proc(**kw) -> dict:
    return ProcedureSpec(**kw).model_dump(mode="json")


def residual_cnn() -> None:
    g = GraphBuilder()
    g.modules.append(residual_block(8))
    g.input("images", ["N", 3, 64, 64])
    g.node("stem", "pytorch.nn.conv2d", out_channels=8, padding=[1, 1])
    g.node("stem_act", "pytorch.nn.relu")
    g.instance("res1", "residual_block", channels=8)
    g.instance("res2", "residual_block", channels=8)
    g.node("pool", "pytorch.nn.adaptive_avg_pool2d", output_size=[1, 1])
    g.node("flatten", "pytorch.nn.flatten")
    g.node("fc", "pytorch.nn.linear", out_features=10)
    g.chain("images", "stem", "stem_act")
    g.wire("stem_act", "res1.x")
    g.wire("res1.y", "res2.x")
    g.wire("res2.y", "pool.input")
    g.chain("pool", "flatten", "fc")
    graph = g.build()
    graph.training = proc(epochs=3, data=DataSpec(kind="image_folder", path="examples/data/shapes10", image_size=(64, 64), batch_size=16, shuffle=True),
                          optimizer=OptimizerSpec(kind="adam", lr=0.003), clip=ClipSpec(kind="norm", max_norm=5.0),
                          validation=ValidationSpec(), checkpoint=CheckpointSpec(every=Cadence(unit="epoch", n=1), keep_last=2))
    write("residual_cnn", graph, "A residual block built without code (module 'residual_block', two clone instances res1 and res2), inside the Phase 1 shapes10 CNN. "
          "Open res1 to see and edit the module. The training procedure uses the generated shapes10 images (run examples/make_shapes10.py first).", True)


def shared_encoder() -> None:
    enc = ModuleBuilder("encoder", "1.0.0", "Two-layer encoder: tanh(W2 tanh(W1 x))")
    x = enc.in_("x", [None, 4])
    enc.node("fc1", "tensor.dense", out_features=8)
    enc.node("a1", "tensor.tanh")
    enc.node("fc2", "tensor.dense", out_features=8)
    enc.node("a2", "tensor.tanh")
    enc.wire(x, "fc1.input")
    enc.chain("fc1", "a1", "fc2", "a2")
    enc.out("z", "a2.output", [None, 8])
    g = GraphBuilder()
    g.modules.append(enc.build())
    g.input("left", ["N", 4])
    g.input("right", ["N", 4])
    g.instance("enc_left", "encoder")
    g.instance("enc_right", "encoder", share="enc_left")
    g.node("diff", "tensor.sub")
    g.node("dist", "tensor.abs")
    g.node("head", "tensor.dense", out_features=1)
    g.wire("left", "enc_left.x")
    g.wire("right", "enc_right.x")
    g.wire("enc_left.z", "diff.a")
    g.wire("enc_right.z", "diff.b")
    g.chain("diff", "dist", "head")
    write("shared_encoder", g.build(), "A Siamese-style model: enc_right is an instance that SHARES its parameters with enc_left (parameter identity, not a copy); the card shows the shared marker. "
          "Gradients from both call sites accumulate on the same tensors.", False)


def masked_loss() -> None:
    g = GraphBuilder()
    g.modules.append(masked_weighted_loss("valid_count"))
    g.input("x", ["N", 4])
    g.node("fc", "tensor.dense", out_features=2)
    g.wire("x", "fc.input")
    graph = g.build()
    graph.training = proc(epochs=8, data=DataSpec(kind="synthetic_regression", features=4, outputs=2, n_train=256, n_val=64, batch_size=32),
                          loss=LossSpec(kind="module", module="masked_weighted_mse_valid_count", ports={"pred": "output", "target": "y", "mask": "mask", "weights": "weights"}),
                          optimizer=OptimizerSpec(kind="sgd", lr=0.05, momentum=0.9), watch=WatchSpec(param="fc.weight", index=[0, 0]))
    write("masked_loss", graph, "A linear model trained with a visual custom loss: module 'masked_weighted_mse_valid_count' (subtract, square, mask, weight, sum, divide by the number of VALID elements) "
          "composed from primitives and used as the training loss. The data is SYNTHETIC: targets with about 25% missing values (mask) and per-target weights.", True)


def transformer() -> None:
    graph = transformer_classifier(vocab=12, seq_len=8, d_model=16, heads=2, d_ff=32, layers=2)
    graph.training = proc(epochs=10, seed=1, data=DataSpec(kind="synthetic_sequence", seq_len=8, vocab=12, n_train=512, n_val=128, batch_size=32),
                          optimizer=OptimizerSpec(kind="adam", lr=0.005), clip=ClipSpec(kind="norm", max_norm=1.0),
                          scheduler=SchedulerSpec(kind="cosine", t_max=160, timing="optimizer_step"),
                          checkpoint=CheckpointSpec(every=Cadence(unit="epoch", n=2), keep_last=3), early_stopping=EarlyStopSpec(patience=5),
                          watch=WatchSpec(param="head.weight", index=[0, 0]))
    write("transformer_sequence", graph, "A 2-layer post-LN transformer encoder from primitives (multi-head self-attention with Q/K/V projections, scaled scores, padding mask, softmax, output "
          "projection, residual + LayerNorm, GELU MLP) classifying SYNTHETIC token sequences: class 1 if token 1 occurs more often than token 2 (id 0 = padding). "
          "Use the Attention inspector on enc1/attn after a run.", True)


def code_block() -> None:
    d = CodeBlockDef(id="softsign", version="1.0.0", description="x / (1 + |x|), written in torch so it is differentiable", inputs=[CodeIO(name="x", shape=["N", 8])],
                     outputs=[CodeIO(name="y", same_as="x")], differentiable=True, randomness="none",
                     source='import torch\n\n\ndef run(x):\n    """softsign activation"""\n    return {"y": x / (1 + x.abs())}\n',
                     fixtures=[{"name": "known values", "inputs": {"x": {"values": [[0.0, 1.0, -1.0, 3.0, -3.0, 0.5, -0.5, 9.0]]}},
                                "expect": {"y": {"values": [[0.0, 0.5, -0.5, 0.75, -0.75, 1 / 3, -1 / 3, 0.9]], "atol": 1e-6}}},
                               {"name": "random batch", "inputs": {"x": {"shape": [4, 8], "seed": 1}}}])
    g = GraphBuilder()
    g.input("x", ["N", 4])
    g.node("fc1", "tensor.dense", out_features=8)
    g.node("act", "code.block", block="softsign")
    g.node("fc2", "tensor.dense", out_features=2)
    g.wire("x", "fc1.input")
    g.wire("fc1.output", "act.x")
    g.wire("act.y", "fc2.input")
    graph = g.build()
    graph.codeBlocks.append(d)
    graph.training = proc(epochs=6, data=DataSpec(kind="synthetic_regression", features=4, outputs=2, n_train=128, n_val=32, batch_size=32),
                          loss=LossSpec(kind="mse", ports={"pred": "output", "target": "y"}), optimizer=OptimizerSpec(kind="adam", lr=0.02))
    write("code_block_demo", graph, "Optional code: the activation 'act' is a Python block (softsign) with a typed interface, run in an isolated subprocess. It is declared differentiable, so the "
           "layer before it trains. Open the Code tab on the node to edit, test with fixtures, and publish a version. The training data is SYNTHETIC.", True)


def control_flow() -> None:
    step = ModuleBuilder("refine", "1.0.0", "h_next = tanh(W (h + context))")
    h, ctx = step.in_("h", [None, 6]), step.in_("context", [None, 6])
    step.node("mix", "tensor.add")
    step.node("fc", "tensor.dense", out_features=6)
    step.node("act", "tensor.tanh")
    step.wire(h, "mix.a")
    step.wire(ctx, "mix.b")
    step.chain("mix", "fc", "act")
    step.out("h", "act.output", [None, 6])
    pos, neg = ModuleBuilder("branch_exp", "1.0.0", "exp(x)"), ModuleBuilder("branch_neg", "1.0.0", "-x")
    for m, op in ((pos, "tensor.exp"), (neg, "tensor.neg")):
        x = m.in_("x", [None, 6])
        m.node("f", op)
        m.wire(x, "f.input")
        m.out("y", "f.output", [None, 6])
    pos.id, neg.id = "branch_exp", "branch_neg"
    g = GraphBuilder()
    g.modules += [step.build(), pos.build(), neg.build()]
    g.input("h0", ["N", 6])
    g.input("context", ["N", 6])
    g.node("loop", "core.repeat", module="refine", version="1.0.0", count=3, carry=[{"input": "h", "output": "h"}], termination={"kind": "fixed_count"}, share="tied")
    g.node("zero", "tensor.constant", value=0.0)
    g.node("positive", "tensor.compare", op="gt")
    g.node("all_positive", "tensor.any_all", kind="all")
    g.node("pick", "core.select", then={"module": "branch_exp", "version": "1.0.0"}, otherwise={"module": "branch_neg", "version": "1.0.0"})
    g.wire("h0", "loop.h")
    g.wire("context", "loop.context")
    g.wire("loop.h", "positive.a")
    g.wire("zero", "positive.b")
    g.wire("positive", "all_positive.input")
    g.wire("all_positive", "pick.pred")
    g.wire("loop.h", "pick.x")
    write("control_flow", g.build(), "Structured control flow, no cycles: 'loop' applies module 'refine' three times with loop-carried state h and tied parameters (fixed_count termination); "
          "'pick' is a typed select between two branch modules with a scalar bool predicate (both branches are evaluated; the predicate chooses the outputs).", False)


if __name__ == "__main__":
    residual_cnn()
    shared_encoder()
    masked_loss()
    transformer()
    code_block()
    control_flow()
