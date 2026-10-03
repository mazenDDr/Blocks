import importlib.util
from pathlib import Path

import pytest

from graph_core.project_io import load_project

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"


def _load_generator():
    spec = importlib.util.spec_from_file_location("make_shapes10", EXAMPLES / "make_shapes10.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def shapes_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("data") / "shapes10"
    _load_generator().generate(out, per_class=6, seed=0)
    return out


@pytest.fixture
def cnn_project():
    return load_project(EXAMPLES / "reference_cnn.project.json")


@pytest.fixture
def cnn_graph(cnn_project):
    return cnn_project.graph


@pytest.fixture
def mse_graph():
    return load_project(EXAMPLES / "mse_teaching.project.json").graph


def set_config(graph, node_id, **updates):
    graph.node(node_id).config.update(updates)
    return graph
