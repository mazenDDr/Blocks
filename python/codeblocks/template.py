"""Minimal source template generated from a code block's declared interface (VISION 15.5 step 2)."""
from __future__ import annotations

from graph_core.schema import CodeBlockDef


def _ty(spec) -> str:
    shape = "any" if spec.shape is None else "[" + ", ".join("?" if d is None else str(d) for d in spec.shape) + "]"
    return f"{spec.dtype} {shape}"


def generate_template(d: CodeBlockDef) -> str:
    args = ", ".join(i.name for i in d.inputs)
    kw = [f"{c.name}={c.default!r}" for c in d.config] + (["state"] if d.state else [])
    sig = args + (", *, " + ", ".join(kw) if kw else "")
    lines = ["import torch", "", "", f"def run({sig}):", f'    """{d.description or d.id}']
    lines += ["", "    inputs:"] + [f"        {i.name}: {_ty(i)}" for i in d.inputs]
    if d.config:
        lines += ["    config:"] + [f"        {c.name}: {c.type} (default {c.default!r})" for c in d.config]
    if d.state:
        lines += ["    state (dict of tensors, updated in place, persists between calls):"] + [f"        {s.name}: {s.dtype} {s.shape}, initially {s.init}" for s in d.state]
    lines += ["    outputs (return a dict with exactly these names):"] + [
        f"        {o.name}: " + (f"same shape and dtype as {o.same_as}" if o.same_as else _ty(o)) for o in d.outputs]
    lines += [f"    effects: {', '.join(d.effects) or 'none'}; randomness: {d.randomness}; "
              + ("differentiable (use torch operations on the input tensors)" if d.differentiable else "not differentiable (outputs carry no gradient)"), '    """']
    first = d.outputs[0] if d.outputs else None
    if first is not None and first.same_as:
        lines.append(f"    return {{{first.name!r}: {first.same_as}.clone()}}  # replace with your computation")
    else:
        lines.append("    raise NotImplementedError('write the computation here')")
    return "\n".join(lines) + "\n"
