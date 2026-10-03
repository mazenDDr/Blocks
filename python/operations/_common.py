"""Helpers shared by operation modules."""
from __future__ import annotations

import math
from typing import Annotated, Any

from pydantic import BeforeValidator, ConfigDict, Field
from pydantic import BaseModel

from graph_core.types import BATCH, Dim, OpError, TensorType


def _to_pair(v: Any) -> Any:
    if isinstance(v, bool):
        raise ValueError("expected an int or a pair of ints")
    if isinstance(v, int):
        return (v, v)
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return tuple(v)
    raise ValueError("expected an int or a pair of ints")


Pair = Annotated[tuple[int, int], BeforeValidator(_to_pair)]
PosPair = Annotated[tuple[Annotated[int, Field(gt=0)], Annotated[int, Field(gt=0)]], BeforeValidator(_to_pair)]
NonNegPair = Annotated[tuple[Annotated[int, Field(ge=0)], Annotated[int, Field(ge=0)]], BeforeValidator(_to_pair)]


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyConfig(StrictConfig):
    pass


def require_rank(t: TensorType, rank: int, port: str, what: str) -> None:
    if len(t.shape) != rank:
        raise OpError("E_RANK_MISMATCH", f"{what} expects a rank-{rank} input, got rank {len(t.shape)} {list(t.shape)}.", port)


def require_dtype(t: TensorType, dtypes: tuple[str, ...], port: str, what: str) -> None:
    if t.dtype not in dtypes:
        raise OpError("E_PORT_TYPE", f"{what} needs dtype in {list(dtypes)}, got {t.dtype}.", port)


def spatial_out(size: int, k: int, s: int, p: int, d: int, ceil_mode: bool = False) -> int:
    num = size + 2 * p - d * (k - 1) - 1
    out = (math.ceil(num / s) if ceil_mode else num // s) + 1
    if ceil_mode and (out - 1) * s >= size + p:
        out -= 1
    return out


def dim_str(d: Dim) -> str:
    return str(d)


def prod_dims(dims: list[Dim]) -> Dim:
    n = 1
    for d in dims:
        if isinstance(d, str):
            return BATCH
        n *= d
    return n
