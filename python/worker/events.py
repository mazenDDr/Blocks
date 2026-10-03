from __future__ import annotations

import time
from typing import Any

from artifact_store import ArtifactStore


class Emitter:
    """Appends ordered events (run id, graph hash, seq, timestamp, type, node id) to the store."""

    def __init__(self, store: ArtifactStore, run_id: str, graph_hash: str):
        self.store, self.run_id, self.graph_hash = store, run_id, graph_hash
        self.seq = 0

    def emit(self, type_: str, node_id: str | None = None, **data: Any) -> None:
        self.store.append_event(self.run_id, self.seq, time.time(), type_, self.graph_hash, node_id, data)
        self.seq += 1
