"""Train the three labelled SYNTHETIC domain examples in real workers and print recorded metrics.

Generate fixtures first: python examples/make_domain_fixtures.py
Run: python examples/domain_journey.py --workbench .workbench
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "python")]

from artifact_store import ArtifactStore
from graph_core.project_io import load_project
from worker.process import submit_run
from worker.tabular_run import TabularRunConfig

EXAMPLES = {
    "vision": ("vision_segmentation_synthetic", "segmenter"),
    "nlp": ("nlp_token_classification", "tagger"),
    "speech": ("speech_ctc_tones", "ctc"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbench", type=Path, default=Path(".workbench"))
    parser.add_argument("--domain", choices=["all", *EXAMPLES], default="all")
    args = parser.parse_args()
    store = ArtifactStore(args.workbench)
    for domain, (project, node) in EXAMPLES.items():
        if args.domain not in ("all", domain):
            continue
        graph = load_project(Path(__file__).parent / f"{project}.project.json").graph
        handle = submit_run(graph, TabularRunConfig(kind="domain", project_id=project), args.workbench)
        try:
            while handle.is_alive():
                time.sleep(0.1)
        except KeyboardInterrupt:
            handle.cancel()
            handle.process.join()
            raise
        handle.process.join()
        row = store.get_run(handle.run_id)
        if row["status"] != "completed":
            raise RuntimeError(f"{domain}: {row['status']}: {row['error']}")
        output = next(a for a in store.artifacts(handle.run_id, "node_output") if a["meta"]["node"] == node)
        summary = next(a for a in store.artifacts(handle.run_id, "node_summary") if a["meta"]["node"] == node)
        data = json.loads(store.read_artifact(output["sha256"]))
        print(json.dumps({"domain": domain, "synthetic": True, "runId": handle.run_id, "graphHash": row["graph_hash"], "node": node,
                          "summarySha256": summary["sha256"], "metrics": data["metrics"]}), flush=True)


if __name__ == "__main__":
    main()
