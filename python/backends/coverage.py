"""The public coverage ledger (VISION 14.3), generated from the registry and the adapters, never edited by hand.

    python -m backends.coverage --write     # regenerate docs/COVERAGE.md
    python -m backends.coverage --check     # exit 1 when docs/COVERAGE.md is stale (tests/test_coverage_ledger.py does the same)

Every cell is DERIVED:
  execution      the adapter's SUPPORTED_TYPES (keras_spec / jax_spec) and the registry's per-op `backend`; "tested" needs conformance cases that run
                 on that backend (backends/conformance.py, the same table the tests parametrize over) or a reference workload containing the op
  visual depth   `explain()` implemented on the Operation; the editor's Architecture tab (ARCH_DEDICATED is checked against InspectorTabs.tsx by a test)
  inspection     activation capture exists on the executables (forward(capture=True)); probes/debugger recording are PyTorch-only
  tests          counts of conformance cases / workloads, and test files that mention the op id literally (static scan of tests/)
Statuses: tested | experimental (implemented, no covering test found) | unsupported (refused before execution, with its code) | n/a (not a model-graph op)."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

from graph_core import registry
from graph_core.registry import Operation

from . import jax_spec, keras_spec
from .conformance import CASES, SPECIFIC_CASES
from .registry import BACKEND_IDS, INFO, pinned_versions
from .workloads import WORKLOADS

ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = ROOT / "docs" / "COVERAGE.md"
SUPPORTED = {"keras": keras_spec.SUPPORTED_TYPES, "jax": jax_spec.SUPPORTED_TYPES}
# ops with a dedicated schematic in the editor's Architecture tab (apps/editor/src/components/InspectorTabs.tsx); every other model op gets the generic in -> out box
ARCH_DEDICATED = {"pytorch.nn.conv2d", "pytorch.nn.linear"}
# expanded into flat nodes by validation (composite instances, repeat, select); their flat nodes are what each backend checks and lowers
STRUCTURAL = {"core.composite", "core.repeat", "core.select"}
TEMPLATE_NOTE = "Architecture templates (the shipped example projects) and pretrained weights are tracked separately from operations; no pretrained weights are provided."


def _test_files() -> dict[str, set[str]]:
    """op type -> test files that contain the op id as a string literal (static scan of tests/*.py)."""
    types = {o.type for o in registry.all_ops()}
    out: dict[str, set[str]] = {}
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        if f.name == "test_coverage_ledger.py":  # the ledger's own test names ops to check the ledger; it is not coverage of them
            continue
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in types:
                out.setdefault(n.value, set()).add(f.name)
    return out


def _workload_ops() -> dict[str, list[str]]:
    out: dict[str, set[str]] = {}
    for name, build in WORKLOADS.items():
        g = build()
        types = {n.type for n in g.nodes} | {n.type for m in g.modules for n in m.nodes}
        for t in types:
            out.setdefault(t, set()).add(name)
    return {t: sorted(v) for t, v in out.items()}


def _has_explain(op: Operation) -> bool:
    return type(op).explain is not Operation.explain


def _model_row(op: Operation, files, workloads) -> dict:
    cases = [c for c in CASES if c.op == op.type]
    row = {"type": op.type, "graphKind": op.graph_kind, "backend": op.backend, "explain": _has_explain(op),
           "architecture": "dedicated schematic" if op.type in ARCH_DEDICATED else "generic in/out box",
           "testFiles": sorted(files.get(op.type, [])), "workloads": workloads.get(op.type, []), "execution": {}, "inspection": {}, "tests": {}, "restrictions": {}}
    if op.type in STRUCTURAL:
        row["execution"] = {b: "expanded" for b in BACKEND_IDS}
        row["inspection"] = {b: "-" for b in BACKEND_IDS}
        row["tests"] = {b: {"cases": [], "workloads": [], "refusalCases": []} for b in BACKEND_IDS}
        row["restrictions"] = {b: [] for b in BACKEND_IDS}
        return row
    for b in BACKEND_IDS:
        runs = [c.id for c in cases if b not in c.expect]
        refused = [(c.id, c.expect[b]) for c in cases if b in c.expect]
        wl = row["workloads"]
        if op.backend in ("keras", "jax"):  # backend-specific node
            if b == op.backend:
                st = "tested" if SPECIFIC_CASES.get(op.type) else "experimental"
                runs = list(SPECIFIC_CASES.get(op.type, []))
            else:
                st, refused, wl = "unsupported", [("(any)", "E_BACKEND_OP")], []
        elif b == "pytorch":
            st = "tested" if (runs or wl or row["testFiles"]) else "experimental"
        elif op.type not in SUPPORTED[b]:
            st, runs, wl = "unsupported", [], []
            refused = [("(any)", "E_BACKEND_UNSUPPORTED_OP")]
        else:
            st = "tested" if (runs or wl) else "experimental"
        row["execution"][b] = st
        if st == "unsupported":
            row["inspection"][b] = "-"
        else:
            row["inspection"][b] = "activation capture + probes + debugger recording" if b == "pytorch" else "activation capture"
        row["tests"][b] = {"cases": runs, "workloads": wl if st != "unsupported" else [], "refusalCases": [r[0] for r in refused if r[0] != "(any)"]}
        row["restrictions"][b] = sorted({f"{code}" + (f" ({cid})" if cid != "(any)" else "") for cid, code in refused})
    return row


def _other_row(op: Operation, files) -> dict:
    return {"type": op.type, "graphKind": op.graph_kind, "backend": op.backend, "explain": _has_explain(op), "view": getattr(op, "summary_kind", None),
            "testFiles": sorted(files.get(op.type, []))}


def build_ledger() -> dict:
    files, workloads = _test_files(), _workload_ops()
    ops = registry.all_ops()
    model = [_model_row(o, files, workloads) for o in ops if o.graph_kind == "model"]
    other = [_other_row(o, files) for o in ops if o.graph_kind != "model"]
    pins = pinned_versions()
    backends = [{"id": b, "title": INFO[b].title, "role": INFO[b].role, "pinned": {p: pins.get(p) for p in INFO[b].packages}, "layout": INFO[b].layout,
                 "training": INFO[b].training, "init": INFO[b].init} for b in BACKEND_IDS]
    refusals = [{"case": c.id, "op": c.op, "config": c.config, "backend": b, "code": code} for c in CASES for b, code in sorted(c.expect.items())]
    return {"backends": backends, "model": model, "other": other, "refusals": refusals, "conformanceCases": len(CASES), "workloads": sorted(WORKLOADS),
            "templateNote": TEMPLATE_NOTE}


# ------------------------------------------------------------------------------------------------ markdown
def _short(t: str) -> str:
    return f"`{t}`"


def render_markdown(L: dict | None = None) -> str:
    L = L or build_ledger()
    o: list[str] = []
    w = o.append
    w("# Coverage ledger")
    w("")
    w("<!-- GENERATED by `python -m backends.coverage --write` from the operation registry, the backend adapters and tests/. Do not edit by hand: "
      "tests/test_coverage_ledger.py fails when this file is stale. -->")
    w("")
    w("This is the public coverage ledger of VISION 14.3: for each operation and backend, whether it executes, how deeply it is explorable, whether its "
      "runtime values can be inspected, and which tests cover it. It is a maintained program, not a claim of 'everything the libraries can do'.")
    w("")
    w("Statuses: **tested** (a test that runs it exists), **experimental** (implemented, no covering test found), **unsupported** (refused before execution with the "
      "stable code shown), **expanded** (structural blocks are expanded into flat nodes before lowering; the flat nodes are what each backend checks). 'Tested' means the cases/workloads named below run in `pytest -q`; where a backend is not installed they skip with its reason.")
    w("")
    w("## Backends and pinned versions")
    w("")
    w("| Backend | Role | Pinned (python/requirements.txt) | Layout | Training | Initialization |")
    w("|---|---|---|---|---|---|")
    for b in L["backends"]:
        pins = ", ".join(f"{p} {v}" for p, v in b["pinned"].items())
        w(f"| {b['title']} (`{b['id']}`) | {b['role']} | {pins} | {b['layout']} | {b['training']} | {b['init']} |")
    w("")
    w(f"Conformance cases: {L['conformanceCases']} single-operation cases (`python/backends/conformance.py`), each run against a handwritten native PyTorch reference on every backend; "
      f"workloads: {', '.join(f'`{x}`' for x in L['workloads'])}.")
    w("")
    w("## Model-graph operations: execution support by backend")
    w("")
    w("`pytorch.nn.*` / `pytorch.loss.*` / `core.*` ids are the historical ids of the portable operations (renaming would break saved projects); whether another backend can run one is decided "
      "per node by its compatibility report, not by the id's prefix. Backend-specific nodes are bound to one backend.")
    w("")
    w("| Operation | Scope | PyTorch | Keras | JAX |")
    w("|---|---|---|---|---|")
    for r in L["model"]:
        scope = "structural" if r["type"] in STRUCTURAL else f"{r['backend']}-specific" if r["backend"] in ("keras", "jax") else ("portable subset" if r["type"] in SUPPORTED["keras"] and r["type"] in SUPPORTED["jax"] else "PyTorch only")
        w(f"| {_short(r['type'])} | {scope} | " + " | ".join(r["execution"][b] for b in BACKEND_IDS) + " |")
    w("")
    w("## Model-graph operations: visual depth and runtime inspection")
    w("")
    w("| Operation | Equation / parameter formula (Explain tab) | Architecture view | Inspection: PyTorch | Keras | JAX |")
    w("|---|---|---|---|---|---|")
    for r in L["model"]:
        w(f"| {_short(r['type'])} | {'yes' if r['explain'] else 'no'} | {r['architecture']} | " + " | ".join(r["inspection"][b] for b in BACKEND_IDS) + " |")
    w("")
    w("Activation capture on Keras/JAX returns every node's value in graph layout (NCHW) after the recorded transposes; probes (`diag.*`), the debugger and recorded reruns are PyTorch-only.")
    w("")
    w("## Model-graph operations: tests")
    w("")
    w("| Operation | PyTorch | Keras | JAX | Test files mentioning the id |")
    w("|---|---|---|---|---|")

    def cell(t):
        parts = []
        if t["cases"]:
            parts.append(f"{len(t['cases'])} conformance")
        if t["refusalCases"]:
            parts.append(f"{len(t['refusalCases'])} refusal")
        if t["workloads"]:
            parts.append(f"{len(t['workloads'])} workload")
        return ", ".join(parts) or "-"

    for r in L["model"]:
        w(f"| {_short(r['type'])} | " + " | ".join(cell(r["tests"][b]) for b in BACKEND_IDS) + f" | {len(r['testFiles'])} |")
    w("")
    w("## Known restrictions (refused before execution)")
    w("")
    w("| Operation | Case | Backend | Stable code |")
    w("|---|---|---|---|")
    for x in L["refusals"]:
        w(f"| {_short(x['op'])} | `{x['case']}` | {x['backend']} | `{x['code']}` |")
    for r in L["model"]:
        for b in BACKEND_IDS:
            if r["execution"][b] == "unsupported":
                w(f"| {_short(r['type'])} | (every configuration) | {b} | `{r['restrictions'][b][0].split(' ')[0]}` |")
    w("")
    w("Also refused on JAX: any float64 tensor (`E_BACKEND_UNSUPPORTED_DTYPE`; JAX would silently downcast). The JAX adapter lowers `int64` labels to `int32` and says so in the report.")
    w("")
    w("## Operations of other graph kinds")
    w("")
    w("Tabular, agent and reinforcement-learning operations run on their own native libraries (scikit-learn, SciPy, LangGraph, Gymnasium + PyTorch); the TensorFlow/JAX backends apply to model graphs only. "
      "Their depth is the explainer / view named below.")
    w("")
    w("| Operation | Graph kind | Native backend | Explain | Dedicated view | Test files mentioning the id |")
    w("|---|---|---|---|---|---|")
    for r in L["other"]:
        w(f"| {_short(r['type'])} | {r['graphKind']} | {r['backend']} | {'yes' if r['explain'] else 'no'} | {r['view'] or '-'} | {len(r['testFiles'])} |")
    w("")
    w("## Templates and pretrained weights")
    w("")
    w(L["templateNote"])
    w("")
    return "\n".join(o)


def main(argv: list[str]) -> int:
    text = render_markdown()
    if "--write" in argv:
        LEDGER_PATH.write_text(text)
        print(f"wrote {LEDGER_PATH}")
        return 0
    if "--check" in argv:
        if not LEDGER_PATH.exists() or LEDGER_PATH.read_text() != text:
            print("docs/COVERAGE.md is stale: run `python -m backends.coverage --write`")
            return 1
        print("docs/COVERAGE.md is current")
        return 0
    print(json.dumps(build_ledger(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
