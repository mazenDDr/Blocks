"""Visual query model for PostgreSQL and its compiler to parameterized SQL.

The compiler is pure (no connection needed): identifiers are quoted with the PostgreSQL rule (double quotes, `"` doubled), every
value is a bound parameter with an explicit cast (typed comparison), operators/functions come from fixed whitelists. The compiled SQL
shown in the inspector is exactly the text that is executed."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .errors import SourceError

ALIAS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
CAST = {"int": "bigint", "float": "double precision", "string": "text", "bool": "boolean", "date": "date", "timestamp": "timestamp"}
MAX_LIMIT = 1_000_000


class _S(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class TableRef(_S):
    schema_: str = Field("public", alias="schema", min_length=1, max_length=63)
    name: str = Field(..., min_length=1, max_length=63)


class ColRef(_S):
    table: str = Field(..., description="alias of the table the column belongs to")
    column: str = Field(..., min_length=1, max_length=63)


class Typed(_S):
    type: Literal["int", "float", "string", "bool", "date", "timestamp"]
    value: Any


class Filter(_S):
    column: ColRef
    op: Literal["=", "!=", "<", "<=", ">", ">=", "in", "not_in", "is_null", "is_not_null", "like"]
    value: Typed | None = None
    values: list[Typed] | None = None


class JoinKey(_S):
    left: ColRef
    right: ColRef


class Join(_S):
    table: TableRef
    alias: str
    type: Literal["inner", "left"] = "inner"
    on: list[JoinKey] = Field(..., min_length=1)


class OutCol(_S):
    table: str
    column: str
    alias: str | None = None


class Agg(_S):
    fn: Literal["count", "count_distinct", "sum", "avg", "min", "max"]
    column: ColRef | None = None
    alias: str


class OrderItem(_S):
    by: str = Field(..., description="name of an output column")
    direction: Literal["asc", "desc"] = "asc"


class QuerySpec(_S):
    base: TableRef
    base_alias: str = "t"
    columns: list[OutCol] = Field(default_factory=list)
    filters: list[Filter] = Field(default_factory=list)
    joins: list[Join] = Field(default_factory=list)
    group_by: list[ColRef] = Field(default_factory=list)
    aggregates: list[Agg] = Field(default_factory=list)
    order_by: list[OrderItem] = Field(default_factory=list)
    limit: int = Field(1000, ge=1, le=MAX_LIMIT)


def q(name: str) -> str:
    if not name or "\x00" in name or len(name) > 63:
        raise SourceError("E_SRC_QUERY_INVALID", f"Invalid identifier {name!r}.")
    return '"' + name.replace('"', '""') + '"'


def typed_value(t: Typed) -> Any:
    v = t.value
    try:
        if t.type == "int":
            if isinstance(v, bool) or (isinstance(v, float) and not v.is_integer()):
                raise ValueError
            return int(v)
        if t.type == "float":
            if isinstance(v, bool):
                raise ValueError
            return float(v)
        if t.type == "string":
            if not isinstance(v, str):
                raise ValueError
            return v
        if t.type == "bool":
            if isinstance(v, bool):
                return v
            if v in ("true", "false"):
                return v == "true"
            raise ValueError
        if t.type == "date":
            return date.fromisoformat(v)
        return datetime.fromisoformat(v)
    except (ValueError, TypeError):
        raise SourceError("E_SRC_TYPE_MISMATCH", f"Value {v!r} is not a valid {t.type}.")


@dataclass
class Compiled:
    sql: str
    params: list[Any]
    param_types: list[str]
    columns: list[str]
    ordered: bool
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"sql": self.sql, "params": [p.isoformat() if isinstance(p, (date, datetime)) else p for p in self.params],
                "paramTypes": self.param_types, "columns": self.columns, "ordered": self.ordered, "warnings": self.warnings,
                "placeholderStyle": "%s (psycopg); values are bound by the driver, never interpolated"}


def compile_query(spec: QuerySpec) -> Compiled:
    aliases: dict[str, TableRef] = {spec.base_alias: spec.base}
    if not ALIAS_RE.match(spec.base_alias):
        raise SourceError("E_SRC_QUERY_INVALID", f"Invalid alias '{spec.base_alias}'.")
    for j in spec.joins:
        if not ALIAS_RE.match(j.alias) or j.alias in aliases:
            raise SourceError("E_SRC_QUERY_INVALID", f"Join alias '{j.alias}' is invalid or already used.")
        aliases[j.alias] = j.table

    def ref(c: ColRef | OutCol) -> str:
        if c.table not in aliases:
            raise SourceError("E_SRC_QUERY_INVALID", f"Unknown table alias '{c.table}' (have {sorted(aliases)}).")
        return f"{q(c.table)}.{q(c.column)}"

    # ---- select list
    out: list[tuple[str, str]] = []  # (expression, output name)
    if spec.aggregates or spec.group_by:
        if spec.columns and {(c.table, c.column) for c in spec.columns} != {(g.table, g.column) for g in spec.group_by}:
            raise SourceError("E_SRC_QUERY_INVALID", "With grouping, the output is the group-by columns plus the aggregates; remove 'columns' or make them equal to 'group_by'.")
        for g in spec.group_by:
            out.append((ref(g), g.column))
        for a in spec.aggregates:
            if not ALIAS_RE.match(a.alias):
                raise SourceError("E_SRC_QUERY_INVALID", f"Invalid aggregate alias '{a.alias}'.")
            if a.column is None:
                if a.fn != "count":
                    raise SourceError("E_SRC_QUERY_INVALID", f"Aggregate '{a.fn}' needs a column.")
                expr = "count(*)"
            elif a.fn == "count_distinct":
                expr = f"count(DISTINCT {ref(a.column)})"
            else:
                expr = f"{a.fn}({ref(a.column)})"
            out.append((expr, a.alias))
    else:
        if not spec.columns:
            raise SourceError("E_SRC_QUERY_INVALID", "Select at least one column (the builder never emits SELECT *).")
        for c in spec.columns:
            out.append((ref(c), c.alias or c.column))
    names = [n for _, n in out]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SourceError("E_SRC_QUERY_INVALID", f"Output column name(s) {dup} are not unique; give them aliases.")

    params: list[Any] = []
    ptypes: list[str] = []

    def bind(v: Any, cast: str) -> str:
        params.append(v)
        ptypes.append(cast)
        return f"%s::{cast}"

    lines = ["SELECT " + ",\n       ".join(f"{e} AS {q(n)}" for e, n in out)]
    lines.append(f"FROM {q(spec.base.schema_)}.{q(spec.base.name)} AS {q(spec.base_alias)}")
    for j in spec.joins:
        on = " AND ".join(f"{ref(k.left)} = {ref(k.right)}" for k in j.on)
        lines.append(f"{'LEFT JOIN' if j.type == 'left' else 'INNER JOIN'} {q(j.table.schema_)}.{q(j.table.name)} AS {q(j.alias)} ON {on}")
    conds = []
    for f in spec.filters:
        c = ref(f.column)
        if f.op in ("is_null", "is_not_null"):
            conds.append(f"{c} IS {'NOT ' if f.op == 'is_not_null' else ''}NULL")
        elif f.op in ("in", "not_in"):
            if not f.values:
                raise SourceError("E_SRC_QUERY_INVALID", f"Filter '{f.op}' on {f.column.column} needs 'values'.")
            types = {v.type for v in f.values}
            if len(types) != 1:
                raise SourceError("E_SRC_TYPE_MISMATCH", "All values of an IN list must have the same type.")
            cast = CAST[types.pop()]
            ph = bind([typed_value(v) for v in f.values], cast + "[]")
            conds.append(f"{c} {'<> ALL' if f.op == 'not_in' else '= ANY'}({ph})")
        else:
            if f.value is None:
                raise SourceError("E_SRC_QUERY_INVALID", f"Filter '{f.op}' on {f.column.column} needs a typed 'value'.")
            if f.op == "like" and f.value.type != "string":
                raise SourceError("E_SRC_TYPE_MISMATCH", "LIKE needs a string value.")
            op = "<>" if f.op == "!=" else ("LIKE" if f.op == "like" else f.op)
            conds.append(f"{c} {op} {bind(typed_value(f.value), CAST[f.value.type])}")
    if conds:
        lines.append("WHERE " + "\n  AND ".join(conds))
    if spec.aggregates or spec.group_by:
        if spec.group_by:
            lines.append("GROUP BY " + ", ".join(ref(g) for g in spec.group_by))
    ordered = bool(spec.order_by)
    if spec.order_by:
        for o in spec.order_by:
            if o.by not in names:
                raise SourceError("E_SRC_QUERY_INVALID", f"ORDER BY refers to '{o.by}', which is not an output column ({names}).")
        lines.append("ORDER BY " + ", ".join(f"{q(o.by)} {o.direction.upper()}" for o in spec.order_by))
    lines.append("LIMIT %s")
    params.append(spec.limit)
    ptypes.append("integer")
    warnings = [] if ordered else ["No ORDER BY: PostgreSQL does not guarantee row order, so the extract's content hash can differ between executions of the same query."]
    return Compiled("\n".join(lines), params, ptypes, names, ordered, warnings)


def join_pairs(spec: QuerySpec) -> list[dict[str, Any]]:
    """Per join: the (left table, right table, key columns) the cardinality report is computed on. Only joins whose left keys all
    belong to one alias can be reported on."""
    aliases = {spec.base_alias: spec.base} | {j.alias: j.table for j in spec.joins}
    res = []
    for j in spec.joins:
        l_aliases = {k.left.table for k in j.on}
        if len(l_aliases) != 1 or any(k.right.table != j.alias for k in j.on):
            res.append({"alias": j.alias, "available": False, "reason": "keys span several tables or the right keys are not on the joined table"})
            continue
        la = l_aliases.pop()
        if la not in aliases:
            raise SourceError("E_SRC_QUERY_INVALID", f"Unknown table alias '{la}'.")
        res.append({"alias": j.alias, "available": True, "leftAlias": la, "left": aliases[la], "right": j.table,
                    "leftKeys": [k.left.column for k in j.on], "rightKeys": [k.right.column for k in j.on], "type": j.type})
    return res
