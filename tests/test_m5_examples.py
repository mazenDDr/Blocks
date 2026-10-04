"""Milestone 5 example projects: deterministic generation, validity, and that the grid-world example actually learns."""
import importlib.util

from conftest import EXAMPLES
from graph_core.project_io import load_project
from graph_core.validate import validate
from rl.train import MemorySink, train
from rl.validate import spec_from_graph

NAMES = ["rl_cartpole_dqn", "rl_gridworld_builder", "unsupervised_cells", "unsupervised_moons"]


def test_generator_is_deterministic_and_matches_committed_files(tmp_path):
    spec = importlib.util.spec_from_file_location("make_m5_examples", EXAMPLES / "make_m5_examples.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(tmp_path)
    for n in NAMES:
        for ext in ("project", "ui"):
            assert (tmp_path / f"{n}.{ext}.json").read_bytes() == (EXAMPLES / f"{n}.{ext}.json").read_bytes(), n


def test_examples_validate_and_are_labelled():
    for n in NAMES:
        p = load_project(EXAMPLES / f"{n}.project.json")
        assert validate(p.graph).ok, n
        assert p.ui.synthetic == n.startswith("unsupervised")


def test_gridworld_example_learns_to_reach_the_goal():
    g = load_project(EXAMPLES / "rl_gridworld_builder.project.json").graph
    spec = spec_from_graph(g)
    spec.trace.enabled = False
    res = train(spec, 0, MemorySink())
    assert res.evals[0].get("successRate") == 0.0 and res.final_eval["successRate"] == 1.0
