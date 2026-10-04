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
    # Explicit parameter sharing (VISION A13): this call site uses the parameter tensors of the node at this path
    # (relative to the scope that contains this node). None = own parameters.
    sharedWith: str | None = None


class Edge(_Open):
    id: str
    kind: str = "tensor"
    from_: Endpoint = Field(alias="from")
    to: Endpoint


class PortSpec(_Open):
    """One named, typed port of a reusable module (its signature). shape entries: int, "N" (batch), another string (a named
    dimension bound consistently across the module's ports), or null (any)."""

    name: str
    dtype: str = "float32"
    shape: list[int | str | None] | None = None
    description: str = ""


class ModuleOutput(_Open):
    name: str
    from_: Endpoint = Field(alias="from")  # inside the module; node "$in" refers to a module input
    dtype: str | None = None
    shape: list[int | str | None] | None = None
    description: str = ""


class ModuleParam(_Open):
    name: str
    default: Any = None
    type: str | None = None  # int | float | str | bool | list; None = not checked
    description: str = ""


class ModuleDef(_Open):
    """A reusable visual function: a subgraph with a typed signature. Instantiated by a `core.composite` node (or
    `core.repeat` / `core.select`). Inside, edges from node "$in" (port = input name) are the module inputs and config values
    `{"$param": name}` are replaced by the instance's argument (or the declared default)."""

    id: str
    version: str = "1.0.0"
    description: str = ""
    inputs: list[PortSpec] = Field(default_factory=list)
    outputs: list[ModuleOutput] = Field(default_factory=list)
    params: list[ModuleParam] = Field(default_factory=list)
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    reduction: dict[str, Any] | None = None  # declared reduction semantics for loss modules


class CodeIO(_Open):
    name: str
    dtype: str = "float32"
    shape: list[int | str | None] | None = None  # or None = same as first input
    same_as: str | None = None  # output only: copy dtype/shape of this input
    description: str = ""


class CodeConfigField(_Open):
    name: str
    type: str = "float"  # int | float | str | bool
    default: Any = 0.0
    description: str = ""


class CodeStateField(_Open):
    name: str
    shape: list[int] = Field(default_factory=lambda: [1])
    dtype: str = "float32"
    init: float = 0.0


class CodeBlockDef(_Open):
    """An optional Python implementation with a declared typed interface (VISION 15.5). The source hash and the pinned
    dependencies are part of the graph, therefore of the semantic hash."""

    id: str
    version: str = "1.0.0"
    description: str = ""
    inputs: list[CodeIO] = Field(default_factory=list)
    outputs: list[CodeIO] = Field(default_factory=list)
    config: list[CodeConfigField] = Field(default_factory=list)
    state: list[CodeStateField] = Field(default_factory=list)
    effects: list[str] = Field(default_factory=list)  # subset of file_read, file_write, network; [] = none declared
    randomness: str = "none"  # none | seeded
    differentiable: bool = False
    dependencies: list[str] = Field(default_factory=list)  # pins like "numpy==2.5.3"
    source: str = ""
    fixtures: list[dict[str, Any]] = Field(default_factory=list)
    limits: dict[str, float] = Field(default_factory=dict)  # cpu_seconds, memory_mb, wall_seconds


class Graph(_Open):
    schemaVersion: str = SCHEMA_VERSION
    graphKind: str = "model"
    backend: str = "pytorch"
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    modules: list[ModuleDef] = Field(default_factory=list)
    codeBlocks: list[CodeBlockDef] = Field(default_factory=list)
    # The declared training procedure of this model graph (a training.spec.ProcedureSpec as JSON): an ordered stage list plus the configuration of each stage.
    training: dict[str, Any] | None = None

    def node(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def to_json(self) -> dict[str, Any]:
        d = self.model_dump(mode="json", by_alias=True)
        # absent features stay absent, so graphs that do not use them keep their existing JSON (and semantic hash)
        for k in ("modules", "codeBlocks", "training"):
            if not d.get(k):
                d.pop(k, None)
        for n in d["nodes"]:
            if n.get("sharedWith") is None:
                n.pop("sharedWith", None)
        return d

    def module(self, module_id: str, version: str | None = None) -> ModuleDef | None:
        for m in self.modules:
            if m.id == module_id and (version is None or m.version == version):
                return m
        return None


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
