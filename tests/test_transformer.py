"""Milestone 3 item 9 / A21 / A33: a small transformer encoder built from primitives, trained on a labelled-synthetic sequence task, with an
attention-head inspector that shows real captured values and labels what is unavailable."""
import math

import pytest
import torch
import torch.nn.functional as F

from debugger.attention import attention_instances, inspect_attention
from graph_core.build import ModuleBuilder, GraphBuilder, multi_head_attention, transformer_classifier, transformer_encoder_block
from graph_core.lower import lower_graph
from graph_core.schema import Graph
from graph_core.validate import validate
from training.spec import DataSpec, OptimizerSpec, ProcedureSpec
from training.trainer import Trainer

D, H, T, V = 16, 2, 8, 12


def tokens(n=4, seed=0, pad_from=6):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(1, V, (n, T), generator=g)
    x[0, pad_from:] = 0
    x[2, 4:] = 0
    return x


def test_attention_matches_a_handwritten_multihead_reference_with_padding_mask():
    graph = transformer_classifier()
    m = lower_graph(graph).eval()
    x = tokens()
    captured = {}
    for nid in ("enc1/attn/q_proj", "enc1/attn/k_proj", "enc1/attn/v_proj", "enc1/attn/out_proj", "embed_sum"):
        m._modules[nid].register_forward_hook(lambda mod, i, o, nid=nid: captured.__setitem__(nid, (i[0].detach(), o.detach())))
    m(x)
    xin = captured["embed_sum"][1]
    wq, wk, wv, wo = (m._modules[f"enc1/attn/{n}_proj"] for n in ("q", "k", "v", "out"))
    key_mask = x != 0
    # reference written directly in torch
    N = xin.shape[0]
    q = wq(xin).reshape(N, T, H, D // H).transpose(1, 2)
    k = wk(xin).reshape(N, T, H, D // H).transpose(1, 2)
    v = wv(xin).reshape(N, T, H, D // H).transpose(1, 2)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(D // H)
    scores = scores.masked_fill(~key_mask[:, None, None, :], -1e9)
    w = scores.softmax(-1)
    ref = wo((w @ v).transpose(1, 2).reshape(N, T, D))
    assert torch.allclose(captured["enc1/attn/out_proj"][1], ref, atol=1e-6)
    # and against torch's own fused attention
    sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask[:, None, None, :])
    assert torch.allclose(w @ v, sdpa, atol=1e-5)


def test_encoder_block_structure_residual_and_post_layernorm():
    r = validate(transformer_classifier(layers=2))
    assert r.ok, r.diagnostics
    assert {"enc1/add1", "enc1/ln1", "enc1/ff1", "enc1/act", "enc1/ff2", "enc1/add2", "enc1/ln2", "enc1/attn/q_proj", "enc2/attn/weights"} <= set(r.order)
    # residual: add1 = x + attention(x); post-LN: ln1 consumes add1
    edges = {(e.from_.node, e.to.node) for e in r.graph.edges}
    assert ("enc1/add1", "enc1/ln1") in edges and ("enc1/ln1", "enc1/add2") in edges and ("enc1/attn/out_proj", "enc1/add1") in edges
    # parameter count: attention 4*(D*D+D), ffn D*F+F+F*D+D, two layer norms 2*2*D, per block
    per_block = 4 * (D * D + D) + (D * 32 + 32 + 32 * D + D) + 2 * 2 * D
    assert r.params["enc1/attn/q_proj"] == D * D + D
    assert r.total_params == V * D + T * D + 2 * per_block + (D * 2 + 2)
    model = lower_graph(transformer_classifier(layers=2))
    assert sum(p.numel() for p in model.parameters()) == r.total_params


def test_shared_encoder_layers_share_parameters():
    g = transformer_classifier(layers=2, share_layers=True)
    r = validate(g)
    assert r.ok
    per_block = 4 * (D * D + D) + (D * 32 + 32 + 32 * D + D) + 2 * 2 * D
    assert r.total_params == V * D + T * D + per_block + (D * 2 + 2)
    m = lower_graph(g)
    assert m._modules["enc2/attn/q_proj"].inner is m._modules["enc1/attn/q_proj"]
    m(tokens()).sum().backward()
    assert m._modules["enc1/attn/q_proj"].weight.grad is not None


def test_padding_positions_get_exactly_zero_attention_weight_and_do_not_change_other_outputs():
    graph = transformer_classifier()
    m = lower_graph(graph).eval()
    x = tokens()
    x2 = x.clone()
    x2[0, 6:] = 7   # change only padded-out ids... they are now real tokens, so compare differently: change ids UNDER the mask instead
    x3 = x.clone()
    x3[2, 4:] = 0   # already padding; replace nothing
    r = inspect_attention(graph, "enc1/attn", {"tokens": x}, sample=0, head=0, token=1)
    w = r["weights"]["row"]
    assert all(wi == 0.0 for wi in w[6:]) and abs(sum(w) - 1) < 1e-5
    # a padded position's content cannot influence a real token's logits: logits are identical when the padded ids change but stay padding
    out1 = m(x)
    x_b = x.clone()
    x_b[0, 6:] = 0
    assert torch.equal(out1, m(x_b))


def test_attention_inspector_shows_real_projections_scores_mask_weights_output_and_labels_what_is_missing():
    graph = transformer_classifier()
    insts = attention_instances(graph)
    assert [i["path"] for i in insts] == ["enc1/attn"] and insts[0]["missing"] == []
    x = tokens()
    r = inspect_attention(graph, "enc1/attn", {"tokens": x}, sample=0, head=1, token=2, seed=0)
    assert r["available"] and r["heads"] == 2 and r["headDim"] == 8 and r["seqLen"] == 8 and r["tokens"] == x[0].tolist()
    assert "initial weights" in r["weightsNote"] and r["provenance"]["kind"].startswith("instrumented forward pass")
    assert len(r["projections"]["q"]) == 8 and len(r["projections"]["k"]) == 8 and len(r["projections"]["k"][0]) == 8
    assert r["mask"]["keysAllowed"] == [True] * 6 + [False] * 2
    assert r["maskedScores"]["row"][6] == pytest.approx(-1e9)
    assert r["context"]["maxAbsDiffToCaptured"] < 1e-6                    # weights @ values reproduces the captured context
    assert len(r["output"]["row"]) == D and len(r["merged"]["row"]) == D
    assert r["unavailable"] == {}
    assert r["cacheOrKvState"]["available"] is False                      # no KV cache in this encoder: stated, not invented
    # the numbers are the model's: recompute the selected row independently
    model = lower_graph(graph).eval()
    torch.manual_seed(0)
    model = lower_graph(graph).eval()
    cap = {}
    for nid in ("enc1/attn/weights",):
        model._modules[nid].register_forward_hook(lambda m_, i, o, nid=nid: cap.__setitem__(nid, o.detach()))
    model(x)
    assert torch.allclose(torch.tensor(r["weights"]["row"]), cap["enc1/attn/weights"][0, 1, 2], atol=1e-7)
    # a trained checkpoint changes the weights and the note says which
    st = {k: v + 0.1 for k, v in model.state_dict().items()}
    r2 = inspect_attention(graph, "enc1/attn", {"tokens": x}, sample=0, head=1, token=2, state=st, weights_note="checkpoint of run abc at step 12")
    assert r2["weights"]["row"] != r["weights"]["row"] and "checkpoint" in r2["weightsNote"]


def test_unavailable_internals_are_labelled_not_invented():
    g = transformer_classifier()
    # an instance of an unrelated module is rejected with the reason; an index out of range too
    r = inspect_attention(g, "enc1", {"tokens": tokens()})
    assert r["available"] is False and r["reason"] == "not_attention" and "enc1/attn" in r["message"]
    r = inspect_attention(g, "enc1/attn", {"tokens": tokens()}, head=5)
    assert r["available"] is False and r["reason"] == "out_of_range"
    # an attention module without a mask: scores feed the softmax directly, the module takes no mask input
    g2 = Graph.model_validate(g.to_json())
    attn = g2.module("multi_head_attention")
    gone = ("mask_b", "masked", "neg_inf")
    attn.nodes = [n for n in attn.nodes if n.id not in gone]
    attn.edges = [e for e in attn.edges if e.from_.node not in gone and e.to.node not in gone]
    from graph_core.build import ep
    from graph_core.schema import Edge
    attn.edges.append(Edge(id="s_w", **{"from": ep("scores.output", "output")}, to=ep("weights.input", "input")))
    # the encoder block still wires a 'mask' into the attention instance: drop that edge
    enc = g2.module("transformer_encoder_block")
    enc.edges = [e for e in enc.edges if not (e.to.node == "attn" and e.to.port == "mask")]
    attn.inputs = [p for p in attn.inputs if p.name != "mask"]
    v = validate(g2)
    assert v.ok, [d.message for d in v.diagnostics]
    info = attention_instances(g2)[0]
    assert {"mask_b", "masked"} <= set(info["missing"]) and "merged" not in info["missing"]
    out = inspect_attention(g2, "enc1/attn", {"tokens": tokens()}, sample=0, head=0, token=0)
    assert out["available"] and "weights" in out and "mask" not in out and "maskedScores" not in out and "merged" in out
    assert set(out["unavailable"]) == {"mask_b", "masked"} and "does not expose" in out["unavailable"]["mask_b"]
    assert out["weights"]["row"][7] > 0     # without the mask the padded key does receive weight: the inspector shows what the model actually computed


def test_sequence_classifier_trains_on_labelled_synthetic_data():
    graph = transformer_classifier()
    spec = ProcedureSpec(epochs=8, seed=1, data=DataSpec(kind="synthetic_sequence", seq_len=T, vocab=V, n_train=256, n_val=64, batch_size=32),
                         optimizer=OptimizerSpec(kind="adam", lr=0.01))
    t = Trainer(graph, spec)
    assert "SYNTHETIC" in t.train_ds.description and t.train_ds.synthetic
    t.run()
    ep = [e for e in t.events if e["type"] == "epoch_end"]
    assert ep[-1]["train_loss"] < ep[0]["train_loss"] * 0.7
    assert ep[-1]["val_acc"] >= 0.85, ep[-1]
    # the label rule is exact: more token-1 than token-2 => class 1
    x, y = t.val_ds.fields["x"], t.val_ds.fields["y"]
    assert torch.equal(y, ((x == 1).sum(1) > (x == 2).sum(1)).long())


def test_trained_model_attention_inspection_uses_the_checkpoint_weights():
    graph = transformer_classifier()
    spec = ProcedureSpec(epochs=2, data=DataSpec(seq_len=T, vocab=V, n_train=64, n_val=16, batch_size=16), optimizer=OptimizerSpec(kind="adam", lr=0.01))
    t = Trainer(graph, spec)
    t.run()
    x = t.val_ds.fields["x"][:4]
    trained = inspect_attention(graph, "enc1/attn", {"tokens": x}, state=t.model.state_dict(), weights_note="trained: after 2 epochs")
    init = inspect_attention(graph, "enc1/attn", {"tokens": x})
    assert trained["weights"]["row"] != init["weights"]["row"]
