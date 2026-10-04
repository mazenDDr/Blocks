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


# ---------------------------------------------------------------------------------------- local test services (real engines)
@pytest.fixture(scope="session")
def pg_service():
    from connectors.localtest import LocalPostgres

    try:
        pg = LocalPostgres()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"local PostgreSQL (pgserver) could not start: {type(e).__name__}: {e}")
    yield pg
    pg.stop()


@pytest.fixture(scope="session")
def s3_service():
    from connectors.localtest import LocalS3

    try:
        s3 = LocalS3()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"local S3-compatible server (moto) could not start: {type(e).__name__}: {e}")
    yield s3
    s3.stop()


@pytest.fixture(scope="session")
def seeded_services(pg_service, s3_service):
    from connectors import synthetic
    from connected_helpers import BUCKET, PLAIN_BUCKET

    synthetic.seed_postgres(pg_service)
    synthetic.seed_s3(s3_service, BUCKET, versioned=True)
    c = s3_service.client()
    c.create_bucket(Bucket=PLAIN_BUCKET)  # deliberately NOT versioned
    c.put_object(Bucket=PLAIN_BUCKET, Key="features/spectra.csv", Body=synthetic.spectra().to_csv(index=False).encode())
    return pg_service, s3_service


@pytest.fixture
def lab(tmp_path, seeded_services, monkeypatch):
    from connected_helpers import Lab

    return Lab(tmp_path, *seeded_services, monkeypatch)


@pytest.fixture
def client_factory():
    """Build a TestClient on a Lab's workbench (the same registry and artifact store the in-process tests use)."""
    from fastapi.testclient import TestClient

    from control.app import create_app

    opened = []

    def make(lab):
        c = TestClient(create_app(lab.wb))
        c.__enter__()
        opened.append(c)
        return c

    yield make
    for c in opened:
        c.__exit__(None, None, None)


def load_domain_generator():
    spec = importlib.util.spec_from_file_location("make_domain_fixtures", EXAMPLES / "make_domain_fixtures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def domain_fixtures():
    """The deterministic SYNTHETIC vision / audio fixtures (gitignored) at their documented location; generated when absent. Returns the generator module."""
    gen = load_domain_generator()
    if not gen.VISION_NPZ.exists():
        gen.write_vision()
    if not gen.AUDIO_NPZ.exists():
        gen.write_audio()
    return gen
