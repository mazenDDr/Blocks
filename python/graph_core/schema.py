"""Pydantic v2 project schema. The spec (Graph) is semantic; layout lives in UiDoc."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"


class _Open(BaseModel):
    # Unknown fields survive load/save (forward compatibility, VISION 16.3).
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class Endpoint(_Open):
    node: str
    port: str


class Node(_Open):
    id: str
    type: str
    version: str = "1.0.0"
    config: dict[str, Any] = Field(default_factory=dict)
    stateRef: str | None = None


class Edge(_Open):
    id: str
    kind: str = "tensor"
    from_: Endpoint = Field(alias="from")
    to: Endpoint


class Graph(_Open):
    schemaVersion: str = SCHEMA_VERSION
    graphKind: str = "model"
    backend: str = "pytorch"
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)

    def node(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


class UiDoc(_Open):
    """Layout only. Never part of the semantic hash."""

    schemaVersion: str = SCHEMA_VERSION
    positions: dict[str, dict[str, float]] = Field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


class ProjectDocument(_Open):
    """A saved project: the semantic graph plus the separate UI document (the API/file contract)."""

    graph: Graph
    ui: UiDoc | None = None
