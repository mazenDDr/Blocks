"""Shared small types: symbolic dims, tensor types, diagnostics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union

BATCH = "N"
Dim = Union[int, str]  # int, or the symbolic batch dimension "N"


@dataclass(frozen=True)
class TensorType:
    shape: tuple[Dim, ...]
    dtype: str = "float32"

    def to_json(self) -> dict[str, Any]:
        return {"shape": list(self.shape), "dtype": self.dtype}


@dataclass
class Fix:
    """A suggested, machine-applicable repair: set one config key on one node."""

    label: str
    node: str | None = None
    key: str | None = None
    value: Any = None

    def to_json(self) -> dict[str, Any]:
        return {"label": self.label, "node": self.node, "key": self.key, "value": self.value}


@dataclass
class Diagnostic:
    code: str
    message: str
    severity: Literal["error", "warning"] = "error"
    nodeId: str | None = None
    port: str | None = None
    path: str = ""
    fixes: list[Fix] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "nodeId": self.nodeId,
            "port": self.port,
            "path": self.path,
            "message": self.message,
            "fixes": [f.to_json() for f in self.fixes],
        }


class OpError(Exception):
    """Raised by an operation's infer_shape; the validator attaches node/path."""

    def __init__(self, code: str, message: str, port: str | None = None, fixes: list[Fix] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.port = port
        self.fixes = fixes or []


def fmt_shape(shape: tuple[Dim, ...]) -> str:
    return "[" + ", ".join(str(d) for d in shape) + "]"
