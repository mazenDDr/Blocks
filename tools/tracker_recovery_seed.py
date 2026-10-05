"""Create actual offline tracker exports in an owned SYNTHETIC smoke workbench."""
import argparse
import json
from pathlib import Path

from artifact_store import ArtifactStore
from graph_core.hashing import semantic_hash
from graph_core.project_io import load_project
from tracking.bridge import Bridge, ExportSelection
from worker.tabular_run import TabularRunConfig, run_tabular


def seed(root):
    store = ArtifactStore(root)
    graph = load_project(Path(__file__).resolve().parents[1] / "examples/production_sensors.project.json").graph
    config = TabularRunConfig()
    run = "SYNTHETIC-tracker-recovery"
    store.create_run(run, semantic_hash(graph), config.model_dump())
    store.add_artifact(run, "graph", graph.model_dump_json(by_alias=True).encode(), "complete", None, {})
    if run_tabular(graph, config, store, run) != "completed":
        raise RuntimeError("Native synthetic tracker source training failed.")
    summary = next(a["sha256"] for a in store.artifacts(run, "node_summary") if a["meta"]["node"] == "metrics")
    bridge = Bridge(root)
    exports = []
    for adapter in ("mlflow", "wandb"):
        row = bridge.queue(ExportSelection(runId=run, adapter=adapter, shareMetrics=True, artifactSha256=[summary]))
        row = bridge.sync(row["id"])
        if row["status"] != "confirmed":
            raise RuntimeError(f"Native {adapter} export failed: {row['error']}")
        exports.append(row)
    return {"fixture": "SYNTHETIC sensors; actual native local MLflow and offline W&B", "exports": exports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbench", required=True)
    args = parser.parse_args()
    print(json.dumps(seed(args.workbench)))
