"""Per-node backend compatibility verdicts. Stable codes:

E_BACKEND_UNSUPPORTED_OP            the backend has no implementation of this operation (nothing is substituted)
E_BACKEND_UNSUPPORTED_PADDING_MODE  padding_mode other than zeros where the backend cannot do it
E_BACKEND_UNSUPPORTED_DILATION      dilation the backend cannot combine with the other settings
E_BACKEND_UNSUPPORTED_CONFIG        another setting (named in the reason) has no equivalent on the backend
E_BACKEND_UNSUPPORTED_DTYPE         a dtype the backend would silently change
E_BACKEND_OP                        a backend-specific node on a graph that targets another backend
E_BACKEND_NAME_COLLISION            a node id would collide with a name in the generated native code
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Status = Literal["supported", "converted", "unsupported"]


@dataclass(frozen=True)
class Conversion:
    kind: str  # layout | weight_layout | padding | op_mapping | dtype | init | semantics
    detail: str

    def to_json(self) -> dict:
        return {"kind": self.kind, "detail": self.detail}


@dataclass
class Check:
    status: Status
    conversions: list[Conversion] = field(default_factory=list)
    code: str | None = None
    reason: str | None = None

    def to_json(self) -> dict:
        return {"status": self.status, "conversions": [c.to_json() for c in self.conversions], "code": self.code, "reason": self.reason}


def native(*conv: Conversion) -> Check:
    """Supported; with conversions if any are listed."""
    return Check("converted" if conv else "supported", list(conv))


def refuse(code: str, reason: str) -> Check:
    return Check("unsupported", [], code, reason)
