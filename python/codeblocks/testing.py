"""Test a code block against fixtures inside the application (VISION 15.5 step 4): type/shape checks of every output against the declared
interface, optional expected values, determinism, declared-randomness check, and (for blocks declared differentiable) a directional-derivative
check of the sandbox gradient. Results are cached by the block's identity (source hash + dependencies + interface) and the fixture, so an edit
to the source or the pins can never be answered from a stale result."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from graph_core.composite import code_interface
from graph_core.schema import CodeBlockDef
from operations.code_ops import CodeBlockConfig, CodeBlockModule, match_shape

from .sandbox import CodeBlockError

DT = {"float32": torch.float32, "float64": torch.float64, "int64": torch.int64, "bool": torch.bool}


def make_tensor(spec: dict[str, Any], port: dict[str, Any]) -> torch.Tensor:
    dt = DT[spec.get("dtype", port["dtype"])]
    if "values" in spec:
        return torch.tensor(spec["values"], dtype=dt)
    g = torch.Generator().manual_seed(int(spec.get("seed", 0)))
    shape = spec["shape"]
    if dt == torch.bool:
        return torch.rand(shape, generator=g) > 0.5
    if dt == torch.int64:
        return torch.randint(int(spec.get("low", 0)), int(spec.get("high", 10)), shape, generator=g)
    return torch.randn(shape, generator=g, dtype=torch.float64).to(dt)


def cache_key(defn: CodeBlockDef, fixture: dict[str, Any]) -> str:
    ident = code_interface(defn)
    return hashlib.sha256(json.dumps([ident["identity"], ident["inputs"], ident["outputs"], ident["config"], ident["state"], ident["limits"], fixture, torch.__version__],
                                     sort_keys=True).encode()).hexdigest()


def _module(defn: CodeBlockDef, params: dict[str, Any]) -> CodeBlockModule:
    cfg = CodeBlockConfig(block=defn.id, version=defn.version, params=params, interface=code_interface(defn))
    return CodeBlockModule(cfg)


def _config_values(defn: CodeBlockDef, given: dict[str, Any]) -> dict[str, Any]:
    vals = {c.name: given.get(c.name, c.default) for c in defn.config}
    for k in given:
        if k not in vals:
            raise CodeBlockError("E_CONFIG", f"fixture config '{k}' is not a declared config field")
    return vals


def run_fixture(defn: CodeBlockDef, fixture: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    res: dict[str, Any] = {"name": fixture.get("name", "fixture"), "ok": False, "checks": [], "error": None, "stdout": ""}

    def check(name: str, ok: bool, detail: str = "") -> bool:
        res["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    mod = None
    try:
        ports = {i.name: i.model_dump() for i in defn.inputs}
        inputs = {n: make_tensor(fixture["inputs"][n], ports[n]) for n in ports}
        mod = _module(defn, _config_values(defn, fixture.get("config", {})))
        binding: dict[str, Any] = {}
        for i in defn.inputs:
            msg = match_shape(i.shape, list(inputs[i.name].shape), binding, f"fixture input '{i.name}'")
            if not check(f"input {i.name} matches the declared type", msg is None and str(inputs[i.name].dtype).replace("torch.", "") == i.dtype, msg or ""):
                raise CodeBlockError("E_FIXTURE", msg or f"fixture input '{i.name}' has the wrong dtype")
        args = [inputs[i.name] for i in defn.inputs]
        seed = fixture.get("seed", 0)
        mod.train()
        outs, _ = mod._call(args, seed if defn.randomness == "seeded" else None, [False] * len(args))   # raises E_CODE_OUTPUT_TYPE / undeclared randomness / exceptions
        res["stdout"] = mod.last_stdout
        for o, t in zip(defn.outputs, outs):
            check(f"output {o.name} type {str(t.dtype).replace('torch.', '')}{list(t.shape)} matches the declared interface", True)
        res["outputs"] = {o.name: {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""), "values": t.flatten()[:32].tolist()} for o, t in zip(defn.outputs, outs)}
        for name, exp in (fixture.get("expect") or {}).items():
            got = outs[[o.name for o in defn.outputs].index(name)]
            want = torch.tensor(exp["values"], dtype=got.dtype)
            ok = got.shape == want.shape and torch.allclose(got.double(), want.double(), atol=exp.get("atol", 1e-6), rtol=exp.get("rtol", 1e-5)) if got.dtype.is_floating_point else torch.equal(got, want)
            check(f"output {name} equals the expected values", ok, "" if ok else f"got {got.flatten()[:8].tolist()}")
        # determinism: same inputs (and seed) give the same outputs
        outs2, _ = mod._call(args, seed if defn.randomness == "seeded" else None, [False] * len(args))
        if not defn.state:
            check("deterministic for equal inputs" + (" and seed" if defn.randomness == "seeded" else ""), all(torch.equal(a, b) for a, b in zip(outs, outs2)))
        check("random numbers are used only if declared" if defn.randomness == "none" else f"randomness declared: {defn.randomness}", True)
        if defn.differentiable:
            res["checks"] += _grad_check(defn, mod, args, seed)
        elif any(t.dtype.is_floating_point for t in args):
            check("non-differentiable boundary: outputs carry no gradient", all(not o.requires_grad for o in outs))
        res["ok"] = all(c["ok"] for c in res["checks"])
    except CodeBlockError as e:
        res["error"] = e.to_json()
        res["stdout"] = e.stdout or res["stdout"]
        expect_err = (fixture.get("expect_error") or {}).get("code")
        if expect_err:
            res["ok"] = e.code == expect_err
            check(f"raises {expect_err}", e.code == expect_err, e.message)
    finally:
        if mod is not None:
            mod.close()
    res["seconds"] = time.perf_counter() - t0
    return res


def _grad_check(defn: CodeBlockDef, mod: CodeBlockModule, args: list[torch.Tensor], seed) -> list[dict[str, Any]]:
    """Gradient from the sandbox (torch.autograd on the block's own torch operations) vs a central finite difference along a random direction."""
    out: list[dict[str, Any]] = []
    sd = seed if defn.randomness == "seeded" else None
    leaf = [a.clone().requires_grad_(a.dtype.is_floating_point) for a in args]
    try:
        outs = mod(*leaf)
    except CodeBlockError as e:
        return [{"name": "gradient flows through torch operations", "ok": False, "detail": e.message}]
    outs = outs if isinstance(outs, tuple) else (outs,)
    loss = sum((o.double() * (1 + torch.arange(o.numel(), dtype=torch.float64).reshape(o.shape) * 0.01)).sum() for o in outs if o.dtype.is_floating_point)
    loss.backward()
    g = [a.grad for a in leaf]
    out.append({"name": "gradient flows through torch operations", "ok": any(x is not None for x in g), "detail": ""})
    rng = torch.Generator().manual_seed(7)
    direction = [torch.randn(a.shape, generator=rng).to(a.dtype) if a.dtype.is_floating_point else None for a in args]
    ana = sum(float((gi.double() * d.double()).sum()) for gi, d in zip(g, direction) if gi is not None and d is not None)
    eps = 1e-3

    def f(sign):
        moved = [(a.double() + sign * eps * d.double()).to(a.dtype) if d is not None else a for a, d in zip(args, direction)]
        os_, _ = mod._call(moved, sd, [False] * len(moved))
        return float(sum((o.double() * (1 + torch.arange(o.numel(), dtype=torch.float64).reshape(o.shape) * 0.01)).sum() for o in os_ if o.dtype.is_floating_point))

    num = (f(+1) - f(-1)) / (2 * eps)
    ok = abs(ana - num) <= 1e-2 * max(1.0, abs(num))
    out.append({"name": "sandbox gradient agrees with a finite difference along a random direction (float32, tol 1e-2)", "ok": ok, "detail": f"autograd {ana:.6g} vs numeric {num:.6g}"})
    return out


def check_block(defn: CodeBlockDef, cache_dir: str | Path | None = None, use_cache: bool = True) -> dict[str, Any]:
    results = []
    for fx in defn.fixtures:
        key = cache_key(defn, fx)
        path = Path(cache_dir) / f"{key}.json" if cache_dir else None
        if use_cache and path is not None and path.exists():
            r = json.loads(path.read_text())
            r["cached"] = True
        else:
            r = run_fixture(defn, fx)
            r["cached"] = False
            transient = (r.get("error") or {}).get("code") in ("E_CODE_TIMEOUT", "E_CODE_CRASH", "E_CODE_CPU_LIMIT")
            if path is not None and not transient:   # a failure caused by the machine is not worth remembering
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(r))
        r["cacheKey"] = key
        results.append(r)
    return {"block": defn.id, "version": defn.version, "sourceSha256": code_interface(defn)["sourceSha256"], "identity": code_interface(defn)["identity"],
            "ok": bool(results) and all(r["ok"] for r in results), "fixtures": results}
