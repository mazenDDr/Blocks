"""Declarative parts of an `agent` graph: the typed state schema with per-field reducers, visual predicates, templates, limits.

Everything here is data (JSON in the project) plus small pure evaluators. No Python source is ever pasted by the user:
a reducer is chosen from a fixed visual set, a route condition is a predicate tree built from field / operator / value pickers,
and a template is text with `{field}` placeholders (not str.format, so no attribute access)."""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

START, END = "START", "END"
RESERVED_NODE_IDS = {START, END}

# --------------------------------------------------------------------------------------------- state schema
FIELD_TYPES = ("text", "integer", "number", "boolean", "list", "object", "messages", "documents", "records", "json")
LISTY = ("list", "messages", "documents", "records")
NUMERIC = ("integer", "number")
REDUCERS = ("replace", "append", "append_unique", "add", "max", "min", "merge", "keep_last_n")
REDUCER_DOC = {
    "replace": "The last write wins. Two parallel branches writing the same field in one step is a conflict (rejected at validation).",
    "append": "New items are appended to the existing list. Parallel writes are all kept.",
    "append_unique": "Appended, skipping items equal to one already present.",
    "add": "Numbers are summed; text is concatenated; lists are joined.",
    "max": "Keeps the larger value.",
    "min": "Keeps the smaller value.",
    "merge": "Objects are merged key by key; the newer value of a key wins.",
    "keep_last_n": "Appended, then only the most recent N items are kept (N = reducer parameter).",
}
# which reducers make sense for which field type (checked at validation, offered by the editor)
REDUCERS_FOR = {
    "text": ("replace", "add"), "integer": ("replace", "add", "max", "min"), "number": ("replace", "add", "max", "min"),
    "boolean": ("replace",), "list": ("replace", "append", "append_unique", "add", "keep_last_n"),
    "messages": ("replace", "append", "append_unique", "keep_last_n"), "documents": ("replace", "append", "append_unique", "keep_last_n"),
    "records": ("replace", "append", "append_unique", "keep_last_n"), "object": ("replace", "merge"), "json": ("replace",),
}
DEFAULTS = {"text": "", "integer": 0, "number": 0.0, "boolean": False, "list": [], "object": {}, "messages": [], "documents": [], "records": [], "json": None}


class Reducer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["replace", "append", "append_unique", "add", "max", "min", "merge", "keep_last_n"] = "replace"
    n: int | None = Field(None, ge=1)  # keep_last_n


class StateField(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    type: Literal["text", "integer", "number", "boolean", "list", "object", "messages", "documents", "records", "json"] = "text"
    reducer: Reducer = Field(default_factory=Reducer)
    default: Any = None
    description: str = ""
    # object fields: the declared shape, so dotted paths (grade.supported) can be type checked
    properties: dict[str, str] = Field(default_factory=dict)
    # turn-scoped fields are reset to their default at the start of every invocation on a thread; thread-scoped ones persist (conversation history)
    scope: Literal["turn", "thread"] = "turn"

    def initial(self) -> Any:
        import copy
        return copy.deepcopy(self.default) if self.default is not None else copy.deepcopy(DEFAULTS[self.type])


def reducer_fn(r: Reducer):
    """The callable handed to LangGraph as the channel's binary operator. None = LastValue (replace)."""
    k = r.kind
    if k == "replace":
        return None
    if k == "append":
        return lambda a, b: list(a or []) + list(b if isinstance(b, list) else [b])
    if k == "append_unique":
        def uniq(a, b):
            out = list(a or [])
            for x in (b if isinstance(b, list) else [b]):
                if x not in out:
                    out.append(x)
            return out
        return uniq
    if k == "add":
        def add(a, b):
            if a is None:
                return b
            if isinstance(a, list):
                return a + list(b if isinstance(b, list) else [b])
            return a + b
        return add
    if k == "max":
        return lambda a, b: b if a is None else max(a, b)
    if k == "min":
        return lambda a, b: b if a is None else min(a, b)
    if k == "merge":
        return lambda a, b: {**(a or {}), **(b or {})}
    if k == "keep_last_n":
        n = r.n or 1
        return lambda a, b: (list(a or []) + list(b if isinstance(b, list) else [b]))[-n:]
    raise ValueError(k)


def apply_reducer(r: Reducer, before: Any, update: Any) -> Any:
    """What the field holds after `update` is written in a step (used by the state diff; LangGraph applies the same function)."""
    fn = reducer_fn(r)
    return update if fn is None else fn(before, update)


# --------------------------------------------------------------------------------------------- limits
class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    maxSteps: int = Field(25, ge=1, le=1000)  # LangGraph recursion limit (supersteps)
    maxModelCalls: int | None = Field(None, ge=1)
    maxTokens: int | None = Field(None, ge=1)  # provider-reported input+output tokens, summed over the run
    maxSeconds: float | None = Field(None, gt=0)
    maxToolCalls: int | None = Field(None, ge=1)


# --------------------------------------------------------------------------------------------- predicates
CMP_OPS = ("==", "!=", "<", "<=", ">", ">=", "contains", "not_contains", "in", "is_empty", "not_empty", "is_true", "is_false", "startswith")
UNARY = ("is_empty", "not_empty", "is_true", "is_false")
NUM_OPS = ("<", "<=", ">", ">=")
SYMBOLS = {"==": "==", "!=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">=", "contains": "contains", "not_contains": "does not contain", "in": "in",
           "is_empty": "is empty", "not_empty": "is not empty", "is_true": "is true", "is_false": "is false", "startswith": "starts with"}
PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def pred_fields(p: dict[str, Any]) -> list[str]:
    """All field paths a predicate reads."""
    out: list[str] = []
    for k in ("all", "any"):
        for q in p.get(k, []) or []:
            out += pred_fields(q)
    if "not" in p and isinstance(p["not"], dict):
        out += pred_fields(p["not"])
    if "field" in p:
        out.append(p["field"])
    if isinstance(p.get("other"), str):
        out.append(p["other"])
    return out


def get_path(ns: dict[str, Any], path: str) -> Any:
    cur: Any = ns
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _operand(ns: dict[str, Any], p: dict[str, Any]) -> tuple[Any, Any]:
    left = get_path(ns, p["field"])
    if p.get("fn") == "len":
        left = len(left) if hasattr(left, "__len__") and left is not None else 0
    right = get_path(ns, p["other"]) if isinstance(p.get("other"), str) else p.get("value")
    return left, right


def eval_predicate(p: dict[str, Any], ns: dict[str, Any], trace: dict[str, Any] | None = None) -> bool:
    """Evaluate a predicate tree against a namespace (the state for routes, a record for memory filters). `trace` collects operand values."""
    if not p or p.get("always") is True:
        return True
    if "all" in p:
        return all([eval_predicate(q, ns, trace) for q in p["all"]])
    if "any" in p:
        return any([eval_predicate(q, ns, trace) for q in p["any"]])
    if "not" in p:
        return not eval_predicate(p["not"], ns, trace)
    left, right = _operand(ns, p)
    if trace is not None:
        key = p["field"] + ("|len" if p.get("fn") == "len" else "")
        trace[key] = left
        if isinstance(p.get("other"), str):
            trace[p["other"]] = right
    op = p.get("op", "==")
    try:
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op in NUM_OPS:
            return left is not None and right is not None and {"<": left < right, "<=": left <= right, ">": left > right, ">=": left >= right}[op]
        if op == "contains":
            return left is not None and right in left
        if op == "not_contains":
            return left is None or right not in left
        if op == "in":
            return left in (right or [])
        if op == "is_empty":
            return left is None or left == "" or (hasattr(left, "__len__") and len(left) == 0)
        if op == "not_empty":
            return not (left is None or left == "" or (hasattr(left, "__len__") and len(left) == 0))
        if op == "is_true":
            return left is True
        if op == "is_false":
            return left is False
        if op == "startswith":
            return isinstance(left, str) and left.startswith(str(right))
    except TypeError:
        return False
    raise ValueError(f"unknown operator {op}")


def render_predicate(p: dict[str, Any]) -> str:
    if not p or p.get("always") is True:
        return "always"
    if "all" in p:
        return "(" + " AND ".join(render_predicate(q) for q in p["all"]) + ")"
    if "any" in p:
        return "(" + " OR ".join(render_predicate(q) for q in p["any"]) + ")"
    if "not" in p:
        return "NOT " + render_predicate(p["not"])
    left = ("len(" + p["field"] + ")") if p.get("fn") == "len" else p["field"]
    op = p.get("op", "==")
    if op in UNARY:
        return f"{left} {SYMBOLS[op]}"
    right = p["other"] if isinstance(p.get("other"), str) else repr(p.get("value"))
    return f"{left} {SYMBOLS[op]} {right}"


# --------------------------------------------------------------------------------------------- templates
VAR_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\}")


def template_vars(t: str) -> list[str]:
    return list(dict.fromkeys(VAR_RE.findall(t or "")))


def render_template(t: str, ns: dict[str, Any]) -> str:
    def sub(m: re.Match) -> str:
        v = get_path(ns, m.group(1))
        if v is None:
            return ""
        if isinstance(v, (dict, list)):
            import json
            return json.dumps(v, ensure_ascii=False)
        return str(v)
    return VAR_RE.sub(sub, t or "")


# --------------------------------------------------------------------------------------------- routes, policies, resources
class Case(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    label: str = ""
    when: dict[str, Any] = Field(default_factory=dict)
    to: str


class Route(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    from_: str = Field(alias="from")
    cases: list[Case] = Field(default_factory=list)
    default: str = END  # where to go when no case matches
    defaultLabel: str = "otherwise"


class Join(BaseModel):
    model_config = ConfigDict(extra="allow")
    node: str
    waitFor: list[str] = Field(default_factory=list)


class EmbeddingsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["ollama", "local_hash"] = "local_hash"
    model: str = "nomic-embed-text"  # ollama only
    dimension: int = Field(256, ge=16, le=4096)  # local_hash only
    normalize: bool = True
    batchSize: int = Field(16, ge=1, le=256)
    baseUrl: str | None = None


class LoaderSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    directory: str = "examples/agent_docs"
    glob: str = "*.txt"
    encoding: str = "utf-8"


class SplitterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: Literal["recursive", "paragraph", "fixed"] = "recursive"
    chunkSize: int = Field(400, ge=20, le=8000)
    chunkOverlap: int = Field(40, ge=0, le=4000)


class IndexSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    description: str = ""
    loader: LoaderSpec = Field(default_factory=LoaderSpec)
    splitter: SplitterSpec = Field(default_factory=SplitterSpec)
    embeddings: EmbeddingsSpec = Field(default_factory=EmbeddingsSpec)
    store: Literal["faiss"] = "faiss"


class PolicyStage(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    op: Literal["retrieve", "filter", "rank", "dedupe", "budget", "summarize"]
    config: dict[str, Any] = Field(default_factory=dict)


class Policy(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    name: str = ""
    description: str = ""
    query: str = ""  # template over the state; its embedding gives each record's relevance
    embeddings: EmbeddingsSpec = Field(default_factory=EmbeddingsSpec)
    stages: list[PolicyStage] = Field(default_factory=list)


class AgentSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    state: list[StateField] = Field(default_factory=list)
    routes: list[Route] = Field(default_factory=list)
    joins: list[Join] = Field(default_factory=list)
    limits: Limits = Field(default_factory=Limits)
    indexes: list[IndexSpec] = Field(default_factory=list)
    policies: list[Policy] = Field(default_factory=list)

    def field(self, name: str) -> StateField | None:
        for f in self.state:
            if f.name == name:
                return f
        return None

    def policy(self, pid: str) -> Policy | None:
        return next((p for p in self.policies if p.id == pid), None)

    def index(self, iid: str) -> IndexSpec | None:
        return next((i for i in self.indexes if i.id == iid), None)


def agent_spec(graph) -> AgentSpec:
    return AgentSpec.model_validate(graph.agent or {})


def resolve_type(spec: AgentSpec, path: str) -> str | None:
    """Static type of a (dotted) state path, or None when unknown."""
    head, *rest = path.split(".")
    f = spec.field(head)
    if f is None:
        return None
    if not rest:
        return f.type
    if f.type == "object" and len(rest) == 1 and rest[0] in f.properties:
        return f.properties[rest[0]]
    return None
