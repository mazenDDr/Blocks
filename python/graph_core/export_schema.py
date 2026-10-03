"""Export the project JSON Schema (graph + separate ui document) for non-Python consumers.

    python -m graph_core.export_schema [path]      # default packages/graph-schema/schema.json"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .schema import ProjectDocument

DEFAULT = Path(__file__).resolve().parents[2] / "packages" / "graph-schema" / "schema.json"


def schema_text() -> str:
    return json.dumps(ProjectDocument.model_json_schema(), indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0]) if argv else DEFAULT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(schema_text())
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
