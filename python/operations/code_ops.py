"""code.block: an optional Python implementation with a declared typed interface (VISION 15.5).

The graph carries the block's definition (graph.codeBlocks) and the node only refers to it, so the source hash, the pinned dependencies and the declared
effects/randomness/differentiability are part of the semantic hash. The source is executed only inside the isolated child process of codeblocks.sandbox."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from pydantic import Field

from codeblocks.sandbox import CodeBlockError, SandboxSession
from graph_core.registry import Operation, register
from graph_core.types import BATCH, Fix, OpError, TensorType

from ._common import StrictConfig

_PY = {"int": int, "float": (int, float), "str": str, "bool": bool}


def match_shape(spec: list | None, actual: tuple | list, binding: dict[str, Any], what: str) -> str | None:
    """Declared shape pattern vs an actual shape. Entries: int exact, "N" batch (any size, but consistent), other str = named dim bound across ports,
    None = any. Works for symbolic ("N") and concrete actual dims."""
    if spec is None:
        return None
    if len(spec) != len(actual):
        return f"{what} must have rank {len(spec)} {list(spec)}, got rank {len(actual)} {list(actual)}"
    for i, (want, got) in enumerate(zip(spec, actual)):
        if want is None:
            continue
        if isinstance(want, str):
            if binding.setdefault(want, got) != got:
                return f"{what}: dimension {i} is '{want}' and was {binding[want]} on another port, got {got}"
        elif want != got:
            return f"{what}: dimension {i} must be {want}, got {got}"
    return None


class CodeBlockConfig(StrictConfig):
    block: str = ""
    version: str = "1.0.0"
    params: dict[str, Any] = Field(default_factory=dict)  # values of the block's declared config fields
    seed: int = 0
    interface: dict[str, Any] | None = None  # filled in from graph.codeBlocks by the expansion step; never edited by hand


def output_types(iface: dict[str, Any], inputs: dict[str, TensorType]) -> dict[str, TensorType]:
    binding: dict[str, Any] = {}
    for spec in iface["inputs"]:
        t = inputs[spec["name"]]
        if t.dtype != spec["dtype"]:
            raise OpError("E_PORT_TYPE", f"input '{spec['name']}' of code block {iface['id']} is declared {spec['dtype']} but receives {t.dtype}. Use tensor.cast.", spec["name"])
        msg = match_shape(spec.get("shape"), t.shape, binding, f"input '{spec['name']}' of code block {iface['id']}")
        if msg:
            raise OpError("E_PORT_TYPE", msg + ".", spec["name"])
    outs: dict[str, TensorType] = {}
    by_name = {s["name"]: s for s in iface["inputs"]}
    for o in iface["outputs"]:
        if o.get("same_as"):
            src = by_name.get(o["same_as"])
            if src is None:
                raise OpError("E_CONFIG", f"output '{o['name']}' is declared same_as '{o['same_as']}', which is not an input.")
            outs[o["name"]] = inputs[o["same_as"]]
            continue
        shape = o.get("shape")
        if shape is None:
            raise OpError("E_CONFIG", f"output '{o['name']}' of code block {iface['id']} needs a declared shape (or same_as an input).")
        dims = []
        for d in shape:
            if d is None:
                raise OpError("E_CONFIG", f"output '{o['name']}' has an unspecified dimension; declare every dimension.")
            if isinstance(d, str) and d != BATCH:
                if d not in binding:
                    raise OpError("E_CONFIG", f"output '{o['name']}' uses dimension name '{d}', which no input binds.")
                d = binding[d]
            dims.append(d)
        outs[o["name"]] = TensorType(tuple(dims), o["dtype"])
    return outs


class _CodeFn(torch.autograd.Function):
    """Differentiable code block: forward and backward both run in the sandbox (backward recomputes the block with the same seed and state)."""

    @staticmethod
    def forward(ctx, module, seed, *inputs):
        ctx.module, ctx.seed = module, seed
        ctx.state_before = {k: v.clone() for k, v in module._state_dict().items()}
        ctx.save_for_backward(*inputs)
        outs, connected = module._call(list(inputs), seed, [t.requires_grad for t in inputs])
        if not any(connected):
            raise CodeBlockError("E_CODE_NOT_DIFFERENTIABLE", f"code block {module.iface['id']} is declared differentiable, but its outputs are not connected to its inputs "
                                 "through torch operations (it probably leaves torch, e.g. numpy or Python scalars). Declare it non-differentiable or use torch ops.")
        return tuple(outs)

    @staticmethod
    def backward(ctx, *grads):
        inputs = ctx.saved_tensors
        m: CodeBlockModule = ctx.module
        rg = list(ctx.needs_input_grad[2:])
        g = m.session.backward(list(inputs), m.params, ctx.state_before, ctx.seed, m.output_names, list(grads), rg)
        return (None, None) + tuple(g)


class CodeBlockModule(nn.Module):
    def __init__(self, cfg: CodeBlockConfig):
        super().__init__()
        self.cfg = cfg
        self.iface = cfg.interface
        self.params = dict(cfg.params)
        self.output_names = [o["name"] for o in self.iface["outputs"]]
        self.differentiable = bool(self.iface["differentiable"])
        self.randomness = self.iface["randomness"]
        self.session = SandboxSession(self.iface, self.iface["source"])
        self.last_stdout = ""
        for s in self.iface.get("state", []):
            self.register_buffer(f"state_{s['name']}", torch.full(s["shape"], s["init"], dtype={"float32": torch.float32, "float64": torch.float64}[s["dtype"]]))

    def __deepcopy__(self, memo):
        new = CodeBlockModule(self.cfg)
        new.load_state_dict(self.state_dict())
        new.train(self.training)
        return new

    def _state_dict(self) -> dict[str, torch.Tensor]:
        return {s["name"]: getattr(self, f"state_{s['name']}") for s in self.iface.get("state", [])}

    def _seed(self) -> int | None:
        if self.randomness == "seeded":
            # drawn from the global torch stream, which checkpoints save: a resumed run feeds the block the same seeds
            return int(torch.randint(0, 2 ** 31 - 1, (1,)).item()) ^ self.cfg.seed
        return None

    def _call(self, inputs: list[torch.Tensor], seed: int | None, rg: list[bool]) -> tuple[list[torch.Tensor], list[bool]]:
        r = self.session.call([t.detach() for t in inputs], self.params, self._state_dict(), seed, self.output_names, rg)
        self.last_stdout = r["stdout"]
        if r["rng_changed"] and self.randomness == "none":
            raise CodeBlockError("E_CODE_UNDECLARED_RANDOMNESS", f"code block {self.iface['id']} consumed random numbers but declares randomness 'none'")
        for k, v in r["state"].items():
            getattr(self, f"state_{k}").copy_(v)
        outs = r["outputs"]
        binding: dict[str, Any] = {}
        for spec, t in zip(self.iface["inputs"], inputs):
            match_shape(spec.get("shape"), list(t.shape), binding, "")
        by_in = {s["name"]: t for s, t in zip(self.iface["inputs"], inputs)}
        for o, t in zip(self.iface["outputs"], outs):
            if o.get("same_as"):
                src = by_in[o["same_as"]]
                if t.shape != src.shape or t.dtype != src.dtype:
                    raise CodeBlockError("E_CODE_OUTPUT_TYPE", f"output '{o['name']}' must have the shape/dtype of input '{o['same_as']}' ({list(src.shape)} {src.dtype}), got {list(t.shape)} {t.dtype}")
                continue
            if str(t.dtype).replace("torch.", "") != o["dtype"]:
                raise CodeBlockError("E_CODE_OUTPUT_TYPE", f"output '{o['name']}' is {str(t.dtype).replace('torch.', '')}, the interface declares {o['dtype']}")
            msg = match_shape(o.get("shape"), list(t.shape), binding, f"output '{o['name']}'")
            if msg:
                raise CodeBlockError("E_CODE_OUTPUT_TYPE", msg)
        return outs, r["connected"]

    def forward(self, *inputs):
        seed = self._seed()
        any_rg = any(t.requires_grad for t in inputs)
        if self.differentiable and any_rg and torch.is_grad_enabled():
            outs = _CodeFn.apply(self, seed, *inputs)
        else:
            outs = tuple(o.detach() for o in self._call(list(inputs), seed, [False] * len(inputs))[0])
        return outs[0] if len(outs) == 1 else outs

    def close(self) -> None:
        self.session.close()


@register
class CodeBlockOp(Operation):
    type = "code.block"
    Config = CodeBlockConfig
    inputs = ()
    outputs = ()

    def input_ports(self, cfg):
        return tuple(i["name"] for i in cfg.interface["inputs"]) if cfg.interface else ()

    def output_ports(self, cfg):
        return tuple(o["name"] for o in cfg.interface["outputs"]) if cfg.interface else ()

    def has_state(self, cfg, params):
        return bool(cfg.interface and cfg.interface.get("state"))

    def resolve(self, cfg, inputs):
        if not cfg.interface:
            return cfg
        merged: dict[str, Any] = {}
        declared = {c["name"]: c for c in cfg.interface["config"]}
        for k in cfg.params:
            if k not in declared:
                raise OpError("E_CONFIG", f"code block {cfg.block} has no config field '{k}' (declared: {sorted(declared)}).")
        for name, c in declared.items():
            v = cfg.params.get(name, c["default"])
            ok = isinstance(v, _PY[c["type"]]) and not (c["type"] != "bool" and isinstance(v, bool))
            if not ok:
                raise OpError("E_CONFIG", f"code block {cfg.block}: config '{name}'={v!r} is not a {c['type']}.", fixes=[Fix(f"Use the default {c['default']!r}", key="params", value={**cfg.params, name: c["default"]})])
            merged[name] = v
        return cfg.model_copy(update={"params": merged})

    def infer_shape(self, cfg, inputs):
        return output_types(cfg.interface, inputs)

    def param_count(self, cfg, inputs):
        return 0

    def lower(self, cfg):
        return CodeBlockModule(cfg)

    def explain(self, cfg, inputs, outputs):
        i = cfg.interface
        n_lines = len(i["source"].splitlines())
        eff = ", ".join(i["effects"]) or "none"
        return {"equation": f"outputs = run(inputs, config) — user code, {n_lines} lines, source sha256 {i['sourceSha256'][:12]}",
                "parameters": {"formula": "0 (state, if any, is a buffer)", "terms": [], "total": 0},
                "executionEffect": f"runs in an isolated subprocess; declared effects: {eff}; randomness: {i['randomness']}; "
                                   + ("differentiable: gradients are computed by torch.autograd inside the sandbox" if i["differentiable"] else
                                      "NOT differentiable: an explicit boundary, outputs carry no gradient"),
                "note": f"dependencies: {', '.join(i['dependencies']) or 'none beyond torch'}. The source is not expanded into a visual graph.",
                "codeBlock": {"id": i["id"], "version": i["version"], "sourceSha256": i["sourceSha256"], "identity": i["identity"], "differentiable": i["differentiable"],
                              "effects": i["effects"], "randomness": i["randomness"], "dependencies": i["dependencies"]}}

    def codegen(self, cfg):
        from graph_core.types import Diagnostic
        from graph_core.validate import ExecutionBlocked

        raise ExecutionBlocked([Diagnostic("E_EXPORT_UNSUPPORTED", "Exporting PyTorch source for a graph with a code block is not implemented yet.", nodeId=None)])
