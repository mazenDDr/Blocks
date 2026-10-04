"""Where connectors find the workbench (connection registry, artifact store) in the control process and in workers."""
from __future__ import annotations

import os
from pathlib import Path

_workbench: Path | None = None


def set_workbench(path: str | Path) -> None:
    global _workbench
    _workbench = Path(path).resolve()


def workbench() -> Path:
    return _workbench or Path(os.environ.get("VOID_WORKBENCH", ".workbench")).resolve()


def registry():
    from .registry import ConnectionRegistry

    return ConnectionRegistry(workbench())
