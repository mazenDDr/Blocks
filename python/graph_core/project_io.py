"""Project save/load. Unknown operations and unknown fields round-trip untouched."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .schema import Graph, UiDoc


@dataclass
class Project:
    graph: Graph
    ui: UiDoc | None = None


def ui_path_for(project_path: str | Path) -> Path:
    p = Path(project_path)
    name = p.name[: -len(".project.json")] if p.name.endswith(".project.json") else p.stem
    return p.with_name(name + ".ui.json")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def save_project(project: Project, path: str | Path) -> None:
    path = Path(path)
    _write_json(path, project.graph.to_json())
    if project.ui is not None:
        _write_json(ui_path_for(path), project.ui.to_json())


def load_project(path: str | Path) -> Project:
    path = Path(path)
    graph = Graph.model_validate(json.loads(path.read_text()))
    ui_file = ui_path_for(path)
    ui = UiDoc.model_validate(json.loads(ui_file.read_text())) if ui_file.exists() else None
    return Project(graph, ui)
