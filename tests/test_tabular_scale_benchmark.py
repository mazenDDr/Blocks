"""The data-scale benchmark harness itself stays runnable (tiny size; the real ladder is in benchmarks/results)."""
import importlib.util
from pathlib import Path


def test_benchmark_runs_the_real_pipeline_at_a_small_size(tmp_path):
    spec = importlib.util.spec_from_file_location("tabular_scale", Path(__file__).resolve().parents[1] / "benchmarks" / "tabular_scale.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = mod.one(2000, tmp_path)
    assert r["runStatus"] == "completed" and r["validationErrors"] == [] and r["staticSourceExact"]
    assert r["artifactsBytes"] > r["csvBytes"] and 0.85 <= r["validationAccuracy"] <= 0.97
