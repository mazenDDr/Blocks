"""Attention-head inspector (VISION 9.4, A33): for one attention instance, one head and one token, the real projections, scaled scores, applied mask,
normalised weights and weighted-value output, taken from an instrumented forward pass of the actual model. Anything the instance does not expose is
reported as unavailable with the reason; nothing is reconstructed or invented.

The module contract it understands (graph_core.build.multi_head_attention): inner nodes q_heads, k_heads, v_heads [N,H,T,dh], scores [N,H,T,T]
(scaled, before masking), mask_b [N,1,1,T] (True = key may be attended), masked, weights, context [N,H,T,dh], merged, out_proj."""
from __future__ import annotations

from typing import Any

import torch

from graph_core.composite import expand
from graph_core.lower import lower_graph
from graph_core.schema import Graph
from graph_core.validate import validate

from .capture import StepCapture

ROLES = ("q_heads", "k_heads", "v_heads", "scores", "mask_b", "masked", "weights", "context", "merged", "out_proj")


def attention_instances(graph: Graph) -> list[dict[str, Any]]:
    """Instances (at any depth) whose module defines the inner nodes needed for the inspector."""
    ex = expand(graph)
    out = []
    for path, inst in ex.instances.items():
        if inst.kind != "core.composite" or not inst.module:
            continue
        have = {m.split("/")[-1] for m in inst.members if m.rsplit("/", 1)[0] == path}
        if {"weights", "scores"} <= have:
            out.append({"path": path, "module": inst.module, "version": inst.version, "roles": sorted(r for r in ROLES if r in have), "missing": sorted(r for r in ROLES if r not in have)})
    return out


def _row(t: torch.Tensor) -> list[float | None]:
    return [None if not torch.isfinite(v) else float(v) for v in t.double().flatten()]


def inspect_attention(graph: Graph, instance: str, inputs: dict[str, torch.Tensor], *, sample: int = 0, head: int = 0, token: int = 0,
                      state: dict[str, torch.Tensor] | None = None, weights_note: str = "initial weights (seeded initialisation; the model has not been trained)",
                      seed: int = 0) -> dict[str, Any]:
    cands = {c["path"]: c for c in attention_instances(graph)}
    if instance not in cands:
        return {"available": False, "reason": "not_attention", "message": f"'{instance}' is not an attention instance this inspector understands (needs inner nodes 'weights' and 'scores'). "
                f"Instances that qualify: {sorted(cands)}"}
    torch.manual_seed(seed)
    model = lower_graph(graph)
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    prefix = instance + "/"
    src = expand(graph).instances[instance].in_src.get("x")
    cap = StepCapture(model, nodes=[n for n in model._modules if n.startswith(prefix)] + ([src[0]] if src else []), with_grads=False).attach()
    with torch.no_grad():
        model(*[inputs[i] for i in model.input_ids])
    cap.detach()
    acts = cap.activations
    first = acts.get(prefix + "q_heads")
    if first is None:
        return {"available": False, "reason": "not_captured", "message": "the instance's inner nodes were not executed"}
    N, H, T, dh = first.shape
    sample, head, token = int(sample), int(head), int(token)
    if not (0 <= sample < N and 0 <= head < H and 0 <= token < T):
        return {"available": False, "reason": "out_of_range", "message": f"sample {sample}/{N}, head {head}/{H}, token {token}/{T} out of range"}
    r: dict[str, Any] = {"available": True, "instance": instance, "module": cands[instance]["module"], "version": cands[instance]["version"], "heads": H, "headDim": dh,
                         "seqLen": T, "batch": N, "sample": sample, "head": head, "token": token, "mode": "eval (dropout off)", "weightsNote": weights_note,
                         "provenance": {"kind": "instrumented forward pass of the actual model", "captured": "forward hooks on the instance's inner nodes (observation only)"},
                         "unavailable": {}}
    for k, v in inputs.items():
        if v.dtype == torch.int64:
            r["tokens"] = v[sample].tolist()
            break

    def get(role: str):
        t = acts.get(prefix + role)
        if t is None:
            r["unavailable"][role] = f"the module does not expose an inner node named '{role}'" if role in cands[instance]["missing"] else "not captured"
        return t

    q, k, v = get("q_heads"), get("k_heads"), get("v_heads")
    if q is not None and k is not None and v is not None:
        r["projections"] = {"q": _row(q[sample, head, token]), "qAll": q[sample, head].tolist(), "k": k[sample, head].tolist(), "v": v[sample, head].tolist(),
                            "meaning": "per-head query of the selected token; keys and values of every token (dimension d_head)"}
    sc, mk, mkd, w, ctx = get("scores"), get("mask_b"), get("masked"), get("weights"), get("context")
    if sc is not None:
        r["scores"] = {"row": _row(sc[sample, head, token]), "matrix": sc[sample, head].tolist(), "meaning": "q·k / sqrt(d_head), before the mask"}
    if mk is not None:
        allowed = mk[sample, 0, 0].tolist() if mk.shape[2] == 1 else mk[sample, 0, token].tolist()
        r["mask"] = {"keysAllowed": allowed, "meaning": "True = this key position may be attended; False = masked (padding)"}
    if mkd is not None:
        r["maskedScores"] = {"row": _row(mkd[sample, head, token])}
    if w is not None:
        r["weights"] = {"row": _row(w[sample, head, token]), "matrix": w[sample, head].tolist(), "rowSum": float(w[sample, head, token].double().sum()),
                        "meaning": "softmax over keys of the masked scores (this is the attention weight, a visualisation of an operation, not an explanation of the output)"}
    if ctx is not None:
        r["context"] = {"row": _row(ctx[sample, head, token]), "meaning": "sum over keys of weight * value, for this head and token"}
        if w is not None and v is not None:
            recomputed = (w[sample, head, token].unsqueeze(-1) * v[sample, head]).sum(0)
            r["context"]["recomputedFromWeightsAndValues"] = _row(recomputed)
            r["context"]["maxAbsDiffToCaptured"] = float((recomputed - ctx[sample, head, token]).abs().max())
    merged, out = get("merged"), get("out_proj")
    if merged is not None:
        r["merged"] = {"row": _row(merged[sample, token]), "meaning": "all heads' contexts concatenated for this token (before the output projection)"}
    if out is not None:
        r["output"] = {"row": _row(out[sample, token]), "meaning": "attention sub-layer output for this token (after the output projection Wo)"}
    x = acts.get(src[0]) if src else None
    if x is not None:
        r["input"] = {"row": _row(x[sample, token]), "producer": f"{src[0]}.{src[1]}"}
    else:
        r["unavailable"]["input"] = "the producer of the attention sub-layer's input was not captured"
    parent = instance.rsplit("/", 1)[0] if "/" in instance else None
    if parent:
        r["residual"] = {"note": "this instance sits inside " + parent + "; its residual add and layer norm are inner nodes there (add1, ln1)"}
    r["cacheOrKvState"] = {"available": False, "reason": "this encoder has no key/value cache: every call recomputes K and V for the whole sequence"}
    return r
