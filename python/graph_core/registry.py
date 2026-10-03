"""Operation contract and registry."""
from __future__ import annotations

from typing import Any, ClassVar

import torch.nn as nn
from pydantic import BaseModel

from .types import OpError, TensorType


class Operation:
    """One operation type. Subclasses are registered once at import time.

    Ports are fixed per op. `Config` is a pydantic model that supplies defaults and validation.
    Configs passed to infer_shape/param_count/lower/explain/codegen are *resolved*:
    `"infer"` placeholders have been replaced with concrete ints by `resolve`.
    """

    type: ClassVar[str]
    version: ClassVar[str] = "1.0.0"
    backend: ClassVar[str] = "pytorch"
    inputs: ClassVar[tuple[str, ...]] = ("input",)
    outputs: ClassVar[tuple[str, ...]] = ("output",)
    Config: ClassVar[type[BaseModel]]

    def resolve(self, cfg: BaseModel, inputs: dict[str, TensorType]) -> BaseModel:
        """Replace "infer" fields using input types. Default: nothing to infer."""
        return cfg

    def infer_shape(self, cfg: Any, inputs: dict[str, TensorType]) -> dict[str, TensorType]:
        raise NotImplementedError

    def param_count(self, cfg: Any, inputs: dict[str, TensorType]) -> int:
        return 0

    def lower(self, cfg: Any) -> nn.Module:
        raise NotImplementedError

    def explain(self, cfg: Any, inputs: dict[str, TensorType], outputs: dict[str, TensorType]) -> dict[str, Any]:
        raise NotImplementedError

    def codegen(self, cfg: Any) -> tuple[str | None, str]:
        """Return (constructor expression or None, forward template).

        The forward template uses `{0}`, `{1}` for input values in port order and `{m}` for
        the module attribute (`self.<node id>`) when a constructor expression exists.
        """
        raise NotImplementedError


_OPS: dict[str, Operation] = {}
_loaded = False


def register(cls: type[Operation]) -> type[Operation]:
    if cls.type in _OPS:
        raise ValueError(f"duplicate operation {cls.type}")
    _OPS[cls.type] = cls()
    return cls


def _ensure_builtins() -> None:
    global _loaded
    if not _loaded:
        _loaded = True
        import operations  # noqa: F401  (registers built-in ops)


def get_op(type_id: str) -> Operation | None:
    _ensure_builtins()
    return _OPS.get(type_id)


def all_ops() -> list[Operation]:
    _ensure_builtins()
    return [_OPS[k] for k in sorted(_OPS)]


__all__ = ["Operation", "OpError", "register", "get_op", "all_ops"]
