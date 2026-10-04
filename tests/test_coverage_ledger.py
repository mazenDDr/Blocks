"""The coverage ledger (VISION 14.3) is generated from code; a stale docs/COVERAGE.md fails the suite."""
import re

from conftest import ROOT
from backends import coverage
from backends.conformance import CASES, SPECIFIC_CASES
from backends.registry import BACKEND_IDS
from graph_core import registry


def test_committed_ledger_is_current():
    committed = (ROOT / "docs" / "COVERAGE.md").read_text()
    assert committed == coverage.render_markdown(), "docs/COVERAGE.md is stale: run `python -m backends.coverage --write` and commit the result"


def test_ledger_lists_every_registered_operation_once():
    L = coverage.build_ledger()
    listed = [r["type"] for r in L["model"]] + [r["type"] for r in L["other"]]
    assert sorted(listed) == sorted(o.type for o in registry.all_ops())
    text = (ROOT / "docs" / "COVERAGE.md").read_text()
    for t in listed:
        assert f"`{t}`" in text


def test_tested_means_cases_or_workloads_exist_and_unsupported_means_a_stable_code():
    L = coverage.build_ledger()
    ids = {c.id for c in CASES}
    for r in L["model"]:
        for b in BACKEND_IDS:
            st, t = r["execution"][b], r["tests"][b]
            if r["type"] in coverage.STRUCTURAL:
                assert st == "expanded"
            elif st == "unsupported":
                assert r["restrictions"][b] and re.match(r"E_BACKEND_[A-Z_]+", r["restrictions"][b][0])
                assert not t["cases"] and not t["workloads"]
            elif b != "pytorch" or t["cases"] or t["workloads"]:
                assert st == "tested" and (t["cases"] or t["workloads"]), (r["type"], b)
            assert set(t["cases"]) <= ids | set(SPECIFIC_CASES.get(r["type"], []))
    portable = {r["type"] for r in L["model"] if r["execution"]["keras"] == "tested" and r["execution"]["jax"] == "tested"}
    assert {"pytorch.nn.conv2d", "pytorch.nn.relu", "pytorch.nn.max_pool2d", "pytorch.nn.adaptive_avg_pool2d", "pytorch.nn.flatten", "pytorch.nn.linear",
            "pytorch.loss.cross_entropy", "core.tensor_input", "core.sub", "core.add", "core.square", "core.sum", "core.mean", "core.scalar_mul"} == portable


def test_architecture_view_claims_match_the_editor_source():
    src = (ROOT / "apps" / "editor" / "src" / "components" / "InspectorTabs.tsx").read_text()
    body = src[src.index("export function ArchitectureTab"):]
    dedicated = set(re.findall(r'node\.type === "([a-z0-9_.]+)"', body))
    assert dedicated == coverage.ARCH_DEDICATED, dedicated


def test_every_backend_specific_op_has_cases_on_its_own_backend():
    for t, names in SPECIFIC_CASES.items():
        assert names and registry.get_op(t).backend in ("keras", "jax")
