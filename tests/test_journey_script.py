"""The connected-data example script runs end to end on its own local services (PostgreSQL via pgserver, S3 API via moto)."""
import subprocess
import sys

import pytest

from conftest import ROOT


def test_connected_journey_script_runs_regression_on_joined_database_and_object_data(tmp_path, pg_service, s3_service):
    # pg_service / s3_service make this a clean skip (with the reason) when the local engines cannot start on this machine
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "connected_journey.py"), "--workbench", str(tmp_path / "wb"), "--no-sweep"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    assert "SYNTHETIC DATA" in out and "regression run" in out and "r2=0.9" in out
    assert "many_to_one, unmatched left 3, unmatched right 1, rows 63" in out
    assert "snapshot assays: postgres" in out and "snapshot spectra: s3" in out
    assert (tmp_path / "wb" / "projects" / "connected_journey.project.json").exists()
    assert "VOID_LAB_S3_SECRET" not in (tmp_path / "wb" / "projects" / "connected_journey.project.json").read_text()
