"""Agent blocks: typed config, declared state reads/writes, visual-first execution.

Each block is a registered operation of graph kind `agent` (control-flow ports `in` / `out`). Executors receive the Runtime (events, model
calls, memory, indexes), the node's config, a state snapshot with defaults filled in, and a NodeCtx (node id, LangGraph step) and
return a partial state update, exactly like a native LangGraph node."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from graph_core.registry import Operation, register

from . import tools as toolmod
from .models import ModelSpec, estimate_tokens
from .policy import run_policy, selection_records, short_term_records
from .spec import (Policy, get_path, pred_fields, reducer_fn, render_template, template_vars)

CHUNK_MARK = re.compile(r"\[([A-Za-z0-9_.\-]+#\d+)\]")


@dataclass
class NodeCtx:
    node_id: str
    step: int | None
    config: Any = None


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------------------------------------ op base
class AgentOp(Operation):
    graph_kind: ClassVar[str] = "agent"
    backend: ClassVar[str] = "langgraph"
    inputs: ClassVar[tuple[str, ...]] = ("in",)
    outputs: ClassVar[tuple[str, ...]] = ("out",)
    in_kinds: ClassVar[dict[str, str]] = {"in": "control"}
    out_kinds: ClassVar[dict[str, str]] = {"out": "control"}
    summary_kind: ClassVar[str] = "agent"
    effects_declared: ClassVar[tuple[str, ...]] = ()

    # statically declared state access (validated against the state schema; recorded in the trace)
    def reads(self, cfg: Any) -> list[str]:
        return []

    def writes(self, cfg: Any) -> list[str]:
        return []

    def templates(self, cfg: Any) -> list[tuple[str, str]]:
        """(location, template text) pairs whose {variables} must be state paths."""
        return []

    def model_specs(self, cfg: Any) -> list[ModelSpec]:
        return []

    def effects(self, cfg: Any) -> list[str]:
        return list(self.effects_declared)

    def check(self, cfg: Any, spec: Any) -> list[tuple[str, str]]:
        """Extra static checks (code, message) given the AgentSpec."""
        return []

    def execute(self, rt: Any, ctx: NodeCtx, cfg: Any, state: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def explain(self, cfg: Any, inputs=None, outputs=None) -> dict[str, Any]:
        return {"summary": self.type, "reads": self.reads(cfg), "writes": self.writes(cfg), "effects": self.effects(cfg)}


def _vars_root(t: str) -> list[str]:
    return [v.split(".")[0] for v in template_vars(t)]


# ================================================================================================ set_state
class Assignment(Strict):
    field: str = ""
    kind: Literal["literal", "copy", "template", "increment", "length", "append_item"] = "literal"
    value: Any = None
    source: str = ""  # copy / length / append_item (dotted state path)
    template: str = ""
    by: float = 1


class SetStateConfig(Strict):
    assignments: list[Assignment] = [Assignment()]


@register
class SetState(AgentOp):
    type = "agent.set_state"
    Config = SetStateConfig

    def reads(self, cfg):
        out = []
        for a in cfg.assignments:
            if a.kind in ("copy", "length", "append_item") and a.source:
                out.append(a.source)
            if a.kind == "increment":
                out.append(a.field)
            out += [v for v in template_vars(a.template)]
        return out

    def writes(self, cfg):
        return [a.field for a in cfg.assignments]

    def templates(self, cfg):
        return [(f"assignments/{i}/template", a.template) for i, a in enumerate(cfg.assignments) if a.kind == "template"]

    def execute(self, rt, ctx, cfg, state):
        out: dict[str, Any] = {}
        for a in cfg.assignments:
            f = rt.spec.field(a.field)
            if a.kind == "literal":
                v = a.value
            elif a.kind == "copy":
                v = get_path(state, a.source)
            elif a.kind == "template":
                v = render_template(a.template, state)
            elif a.kind == "length":
                x = get_path(state, a.source)
                v = len(x) if x is not None else 0
            elif a.kind == "append_item":
                v = [get_path(state, a.source) if a.source else a.value]
            else:  # increment: "add `by` to the field", whatever the reducer is
                by = int(a.by) if f is not None and f.type == "integer" else a.by
                v = by if (f is not None and f.reducer.kind == "add") else (state.get(a.field) or 0) + by
            out[a.field] = v
        return out

    def explain(self, cfg, inputs=None, outputs=None):
        return {"summary": "Writes fields of the state from literals, copies, templates or counters.", "reads": self.reads(cfg), "writes": self.writes(cfg), "effects": []}


# ================================================================================================ prompt
class PromptItem(Strict):
    kind: Literal["template", "memory", "conversation", "documents", "tool_results"] = "template"
    role: Literal["system", "user", "assistant"] = "user"
    template: str = ""  # template: the message text. memory/documents/tool_results: per-item line template
    field: str = ""  # memory / conversation / documents / tool_results: the state field holding the items
    header: str = ""  # memory / documents / tool_results: first line of the combined message


class PromptConfig(Strict):
    items: list[PromptItem] = [PromptItem(kind="template", role="system", template="You are a careful research assistant."),
                              PromptItem(kind="template", role="user", template="{question}")]
    output_field: str = "prompt_messages"


DEFAULT_ITEM_TEMPLATE = {"memory": "- {text}", "documents": "[{chunk_id}] {text}", "tool_results": "{tool}({args}) = {result}", "conversation": ""}


@register
class Prompt(AgentOp):
    type = "agent.prompt"
    Config = PromptConfig

    def reads(self, cfg):
        out = []
        for it in cfg.items:
            if it.kind == "template":
                out += template_vars(it.template)
            elif it.field:
                out.append(it.field)
        return out

    def writes(self, cfg):
        return [cfg.output_field]

    def templates(self, cfg):
        return [(f"items/{i}/template", it.template) for i, it in enumerate(cfg.items) if it.kind == "template"]

    def check(self, cfg, spec):
        out = []
        for i, it in enumerate(cfg.items):
            if it.kind != "template" and not it.field:
                out.append(("E_CONFIG", f"items/{i}: a '{it.kind}' item needs a state field"))
            if it.kind != "template" and it.field:
                f = spec.field(it.field)
                want = {"memory": "records", "conversation": "records", "documents": "documents", "tool_results": "list"}[it.kind]
                if f is not None and f.type != want and not (it.kind == "conversation" and f.type == "messages"):
                    out.append(("E_STATE_TYPE", f"items/{i}: a '{it.kind}' item reads a '{want}' field but '{it.field}' is '{f.type}'"))
        return out

    def execute(self, rt, ctx, cfg, state):
        msgs = render_prompt(cfg, state, ctx.node_id)
        rt.emit("prompt_rendered", ctx.node_id, messages=len(msgs), tokensEstimate=sum(estimate_tokens(m["content"]) for m in msgs),
                inputs=_snapshot(state, self.reads(cfg)), items=[it.kind for it in cfg.items])
        return {cfg.output_field: msgs}

    def explain(self, cfg, inputs=None, outputs=None):
        return {"summary": "Renders ordered message templates, memory selections, retrieved chunks and tool results into the message list a model block will send.",
                "reads": self.reads(cfg), "writes": self.writes(cfg), "effects": []}


def _snapshot(state: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    out = {}
    for f in dict.fromkeys(x.split(".")[0] for x in fields):
        if f in state:
            out[f] = state[f]
    return out


def _message(role: str, parts: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    text, segs = "", []
    for t, src in parts:
        segs.append({"start": len(text), "end": len(text) + len(t), "source": src, "tokensEstimate": estimate_tokens(t)})
        text += t
    return {"role": role, "content": text, "segments": segs}


def render_prompt(cfg: PromptConfig, state: dict[str, Any], node_id: str) -> list[dict[str, Any]]:
    """Rendered message list; each message carries `segments` linking every character range to its source."""
    out: list[dict[str, Any]] = []
    for i, it in enumerate(cfg.items):
        base = {"node": node_id, "item": i}
        if it.kind == "template":
            text = render_template(it.template, state)
            if text.strip():
                out.append(_message(it.role, [(text, {"kind": "prompt_template", **base, "variables": template_vars(it.template)})]))
            continue
        items = state.get(it.field) or []
        if it.kind == "conversation":
            for r in items:
                role = (r.get("metadata", {}).get("role") or r.get("role") or "user")
                text = r.get("text", r.get("content", ""))
                out.append(_message(role if role in ("system", "user", "assistant") else "user", [(text, {"kind": "conversation_message", "recordId": r.get("id"), **base,
                                                                                                     "applicationId": r.get("applicationId")})]))
            continue
        if not items:
            continue
        line_t = it.template or DEFAULT_ITEM_TEMPLATE[it.kind]
        parts: list[tuple[str, dict[str, Any]]] = []
        if it.header:
            parts.append((it.header + "\n", {"kind": "prompt_template", **base, "variables": []}))
        for r in items:
            line = render_template(line_t, {**r, "result": json.dumps(r.get("result")) if isinstance(r.get("result"), (dict, list)) else r.get("result"),
                                            "args": json.dumps(r.get("args")) if isinstance(r.get("args"), dict) else r.get("args")}) + "\n"
            if it.kind == "memory":
                src = {"kind": "memory_record", "recordId": r.get("id"), "store": r.get("store"), "recordKind": r.get("kind"), "applicationId": r.get("applicationId"), **base}
            elif it.kind == "documents":
                src = {"kind": "retrieved_chunk", "chunkId": r.get("chunk_id"), "docId": r.get("doc_id"), "score": r.get("score"), "retrievalId": r.get("retrievalId"),
                       "sourcePath": r.get("source_path"), **base}
            else:
                src = {"kind": "tool_result", "tool": r.get("tool"), "callId": r.get("callId"), **base}
            parts.append((line, src))
        if parts:
            out.append(_message(it.role, parts))
    return out


# ================================================================================================ chat model / structured output
class ChatModelConfig(Strict):
    model: ModelSpec = ModelSpec()
    messages_field: str = "prompt_messages"
    output_field: str = "answer"
    append_to: str = ""  # optional messages/records field that also receives the assistant message
    max_context_note: str = ""


@register
class ChatModel(AgentOp):
    type = "agent.chat_model"
    Config = ChatModelConfig

    def reads(self, cfg):
        return [cfg.messages_field]

    def writes(self, cfg):
        return [cfg.output_field] + ([cfg.append_to] if cfg.append_to else [])

    def model_specs(self, cfg):
        return [cfg.model]

    def effects(self, cfg):
        if cfg.model.provider == "openai_compatible":
            from .openai_compat import is_loopback
            return ["local_runtime"] if is_loopback(cfg.model.base_url) else ["network"]
        return ["network"] if cfg.model.provider != "fixture" and cfg.model.provider != "ollama" else (["local_runtime"] if cfg.model.provider == "ollama" else [])

    def execute(self, rt, ctx, cfg, state):
        res = rt.model_call(ctx, cfg.model, state.get(cfg.messages_field) or [], purpose="chat", messages_field=cfg.messages_field)
        out = {cfg.output_field: res["text"]}
        if cfg.append_to:
            out[cfg.append_to] = [{"id": f"msg:{rt.thread_id}:{res['callId']}", "role": "assistant", "content": res["text"], "ts": time.time(), "text": res["text"],
                                   "source": {"kind": "model_call", "callId": res["callId"], "node": ctx.node_id, "runId": rt.run_id}}]
        return out

    def explain(self, cfg, inputs=None, outputs=None):
        return {"summary": "Sends the message list to the selected model provider (a model invocation, not training) and stores the reply text.", "reads": self.reads(cfg),
                "writes": self.writes(cfg), "effects": self.effects(cfg)}


class SchemaField(Strict):
    name: str = "value"
    type: Literal["text", "integer", "number", "boolean", "enum", "list_of_text"] = "text"
    description: str = ""
    required: bool = True
    choices: list[str] = []


class RetryPolicy(Strict):
    maxRetries: int = Field(2, ge=0, le=10)
    feedback: bool = True  # add the validation error to the retry request


class StructuredOutputConfig(Strict):
    model: ModelSpec = ModelSpec()
    messages_field: str = "prompt_messages"
    output_field: str = "result"
    error_field: str = ""
    schema_fields: list[SchemaField] = [SchemaField()]
    retry: RetryPolicy = RetryPolicy()
    on_failure: Literal["route", "fail"] = "route"  # route: write {} and the error text so a predicate can route; fail: the node fails


def json_schema_of(fields: list[SchemaField]) -> dict[str, Any]:
    props: dict[str, Any] = {}
    for f in fields:
        if f.type == "text":
            p: dict[str, Any] = {"type": "string"}
        elif f.type == "integer":
            p = {"type": "integer"}
        elif f.type == "number":
            p = {"type": "number"}
        elif f.type == "boolean":
            p = {"type": "boolean"}
        elif f.type == "enum":
            p = {"type": "string", "enum": f.choices}
        else:
            p = {"type": "array", "items": {"type": "string"}}
        if f.description:
            p["description"] = f.description
        props[f.name] = p
    return {"type": "object", "properties": props, "required": [f.name for f in fields if f.required], "additionalProperties": False}


def validate_structured(text: str, fields: list[SchemaField]) -> tuple[dict[str, Any] | None, list[str]]:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        obj = json.loads(t)
    except ValueError as e:
        return None, [f"not valid JSON ({str(e)[:80]})"]
    if not isinstance(obj, dict):
        return None, ["the JSON value is not an object"]
    errs = []
    for f in fields:
        if f.name not in obj:
            if f.required:
                errs.append(f"missing required field '{f.name}'")
            continue
        v = obj[f.name]
        ok = {"text": isinstance(v, str), "integer": isinstance(v, int) and not isinstance(v, bool), "number": isinstance(v, (int, float)) and not isinstance(v, bool),
              "boolean": isinstance(v, bool), "enum": isinstance(v, str) and v in f.choices, "list_of_text": isinstance(v, list) and all(isinstance(x, str) for x in v)}[f.type]
        if not ok:
            errs.append(f"field '{f.name}' must be {f.type}" + (f" in {f.choices}" if f.type == "enum" else "") + f", got {json.dumps(v)[:40]}")
    extra = [k for k in obj if k not in {f.name for f in fields}]
    if extra:
        errs.append(f"unexpected fields {extra}")
    return (obj if not errs else None), errs


@register
class StructuredOutput(AgentOp):
    type = "agent.structured_output"
    Config = StructuredOutputConfig

    def reads(self, cfg):
        return [cfg.messages_field]

    def writes(self, cfg):
        return [cfg.output_field] + ([cfg.error_field] if cfg.error_field else [])

    def model_specs(self, cfg):
        return [cfg.model]

    def effects(self, cfg):
        if cfg.model.provider == "openai_compatible":
            from .openai_compat import is_loopback
            return ["local_runtime"] if is_loopback(cfg.model.base_url) else ["network"]
        return ["local_runtime"] if cfg.model.provider == "ollama" else (["network"] if cfg.model.provider == "anthropic" else [])

    def check(self, cfg, spec):
        out = []
        names = [f.name for f in cfg.schema_fields]
        if len(set(names)) != len(names):
            out.append(("E_CONFIG", "schema field names must be unique"))
        for f in cfg.schema_fields:
            if f.type == "enum" and not f.choices:
                out.append(("E_CONFIG", f"enum field '{f.name}' needs choices"))
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", f.name):
                out.append(("E_CONFIG", f"schema field name '{f.name}' is not an identifier"))
        of = spec.field(cfg.output_field)
        if of is not None and of.type == "object" and of.properties:
            decl = {f.name: {"text": "text", "integer": "integer", "number": "number", "boolean": "boolean", "enum": "text", "list_of_text": "list"}[f.type] for f in cfg.schema_fields}
            if decl != of.properties:
                out.append(("E_STATE_TYPE", f"the schema declares {decl} but state field '{cfg.output_field}' declares properties {of.properties}"))
        return out

    def execute(self, rt, ctx, cfg, state):
        schema = json_schema_of(cfg.schema_fields)
        instruction = {"role": "user", "content": "Respond with ONLY a JSON object that matches this JSON Schema, with no other text:\n" + json.dumps(schema),
                       "segments": []}
        instruction["segments"] = [{"start": 0, "end": len(instruction["content"]), "source": {"kind": "structured_output_instruction", "node": ctx.node_id},
                                    "tokensEstimate": estimate_tokens(instruction["content"])}]
        msgs = list(state.get(cfg.messages_field) or []) + [instruction]
        attempts = []
        last_err: list[str] = []
        for attempt in range(cfg.retry.maxRetries + 1):
            res = rt.model_call(ctx, cfg.model, msgs, purpose="structured_output", attempt=attempt + 1, json_schema=schema, validate=lambda t: validate_structured(t, cfg.schema_fields)[1], messages_field=cfg.messages_field)
            obj, errs = validate_structured(res["text"], cfg.schema_fields)
            attempts.append({"attempt": attempt + 1, "callId": res["callId"], "valid": obj is not None, "errors": errs})
            rt.emit("structured_attempt", ctx.node_id, attempt=attempt + 1, callId=res["callId"], valid=obj is not None, errors=errs, maxRetries=cfg.retry.maxRetries)
            if obj is not None:
                out = {cfg.output_field: obj}
                if cfg.error_field:
                    out[cfg.error_field] = ""
                return out
            last_err = errs
            if cfg.retry.feedback:
                fb = {"role": "user", "content": "Your previous reply was rejected: " + "; ".join(errs) + ". Reply again with ONLY the JSON object.", "segments": []}
                fb["segments"] = [{"start": 0, "end": len(fb["content"]), "source": {"kind": "structured_output_retry_feedback", "node": ctx.node_id, "attempt": attempt + 1},
                                   "tokensEstimate": estimate_tokens(fb["content"])}]
                msgs = msgs + [{"role": "assistant", "content": res["text"], "segments": [{"start": 0, "end": len(res["text"]), "source": {"kind": "model_reply", "callId": res["callId"]},
                                                                                         "tokensEstimate": estimate_tokens(res["text"])}]}, fb]
        if cfg.on_failure == "fail":
            raise NodeFailure("E_STRUCTURED_OUTPUT", f"output did not validate after {len(attempts)} attempts: {'; '.join(last_err)}")
        out = {cfg.output_field: {}}
        if cfg.error_field:
            out[cfg.error_field] = f"validation failed after {len(attempts)} attempts: " + "; ".join(last_err)
        return out

    def explain(self, cfg, inputs=None, outputs=None):
        return {"summary": "Asks the model for a JSON object matching the declared schema, validates it here, retries per the retry policy, then routes on failure.",
                "reads": self.reads(cfg), "writes": self.writes(cfg), "effects": self.effects(cfg), "schema": json_schema_of(cfg.schema_fields)}


class NodeFailure(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


# ================================================================================================ embeddings / retrieval
class EmbedTextConfig(Strict):
    from_field: str = "question"
    output_field: str = "embedding_info"
    provider: Literal["ollama", "local_hash"] = "local_hash"
    model: str = "nomic-embed-text"
    dimension: int = Field(256, ge=16, le=4096)
    normalize: bool = True


@register
class EmbedText(AgentOp):
    type = "agent.embed_text"
    Config = EmbedTextConfig

    def reads(self, cfg):
        return [cfg.from_field]

    def writes(self, cfg):
        return [cfg.output_field]

    def effects(self, cfg):
        return ["local_runtime"] if cfg.provider == "ollama" else []

    def execute(self, rt, ctx, cfg, state):
        import math

        from .models import Embedder
        from .spec import EmbeddingsSpec

        e = Embedder(EmbeddingsSpec(provider=cfg.provider, model=cfg.model, dimension=cfg.dimension, normalize=cfg.normalize))
        v = e.embed([str(state.get(cfg.from_field) or "")])[0]
        info = {"identity": e.identity, "label": e.label, "dimension": len(v), "norm": round(math.sqrt(sum(x * x for x in v)), 5), "normalized": cfg.normalize,
                "head": [round(x, 4) for x in v[:8]]}
        rt.emit("embedding", ctx.node_id, **info)
        return {cfg.output_field: info}

    def explain(self, cfg, inputs=None, outputs=None):
        return {"summary": "Embeds one text field and records the embedding identity, dimension and norm.", "reads": self.reads(cfg), "writes": self.writes(cfg), "effects": self.effects(cfg)}


class RetrieveConfig(Strict):
    index: str = ""
    query: str = "{question}"
    k: int = Field(3, ge=1, le=50)
    score_threshold: float | None = None
    output_field: str = "docs"


@register
class Retrieve(AgentOp):
    type = "agent.retrieve"
    Config = RetrieveConfig

    def reads(self, cfg):
        return template_vars(cfg.query)

    def writes(self, cfg):
        return [cfg.output_field]

    def templates(self, cfg):
        return [("query", cfg.query)]

    def check(self, cfg, spec):
        out = []
        if not cfg.index or spec.index(cfg.index) is None:
            out.append(("E_INDEX_UNKNOWN", f"index '{cfg.index}' is not declared in this graph's indexes"))
        return out

    def effects(self, cfg):
        return ["file_read"]

    def execute(self, rt, ctx, cfg, state):
        ispec = rt.spec.index(cfg.index)
        man = rt.ensure_index(ispec)
        q = render_template(cfg.query, state).strip()
        res = rt.indexes.search(ispec, q, cfg.k, cfg.score_threshold)
        from .memory import new_id

        rid = new_id("ret")
        payload = {"id": rid, "runId": rt.run_id, "threadId": rt.thread_id, "node": ctx.node_id, "index": cfg.index, "query": q, "k": cfg.k, "scoreThreshold": cfg.score_threshold,
                   "included": res["documents"], "excluded": res["excluded"], "embedding": res["embedding"], "scoreInterpretation": res["scoreInterpretation"]}
        rt.memory.save_retrieval(rid, payload)
        rt.emit("retrieval", ctx.node_id, retrievalId=rid, index=cfg.index, query=q, k=cfg.k, scoreThreshold=cfg.score_threshold, embedding=res["embedding"]["identity"],
                scoreInterpretation=res["scoreInterpretation"], included=[{k: d[k] for k in ("chunk_id", "score", "rank")} for d in res["documents"]],
                excluded=[{k: d[k] for k in ("chunk_id", "score", "rank", "reason")} for d in res["excluded"][:20]], indexAction=man["action"])
        return {cfg.output_field: [{**d, "retrievalId": rid} for d in res["documents"]]}

    def explain(self, cfg, inputs=None, outputs=None):
        return {"summary": "Similarity search over a persisted FAISS index; returns the top-k chunks above the score threshold with their scores.", "reads": self.reads(cfg),
                "writes": self.writes(cfg), "effects": ["file_read"]}


class CitationsConfig(Strict):
    text_field: str = "answer"
    docs_field: str = "docs"
    output_field: str = "citation_check"


@register
class Citations(AgentOp):
    type = "agent.citations"
    Config = CitationsConfig

    def reads(self, cfg):
        return [cfg.text_field, cfg.docs_field]

    def writes(self, cfg):
        return [cfg.output_field]

    def execute(self, rt, ctx, cfg, state):
        docs = {d.get("chunk_id"): d for d in state.get(cfg.docs_field) or []}
        found = []
        for m in dict.fromkeys(CHUNK_MARK.findall(str(state.get(cfg.text_field) or ""))):
            d = docs.get(m)
            found.append({"marker": f"[{m}]", "chunk_id": m, "valid": d is not None, "doc_id": d.get("doc_id") if d else None, "source_path": d.get("source_path") if d else None,
                          "score": d.get("score") if d else None})
        res = {"citations": found, "count": len(found), "invalid": sum(1 for c in found if not c["valid"]), "allValid": bool(found) and all(c["valid"] for c in found)}
        rt.emit("citations_checked", ctx.node_id, **res)
        return {cfg.output_field: res}

    def explain(self, cfg, inputs=None, outputs=None):
        return {"summary": "Extracts [chunk_id] markers from generated text and checks each against the chunks that were actually retrieved. A marker that is not a retrieved chunk is invalid; nothing is invented.",
                "reads": self.reads(cfg), "writes": self.writes(cfg), "effects": []}


# ================================================================================================ tools
class ToolCallConfig(Strict):
    tool: Literal["calculator", "read_text_file", "write_note"] = "calculator"
    args: dict[str, str] = {"expression": "{question}"}  # argument -> template over the state
    allowed_dir: str = ""  # file tools: the only directory they may touch (default: the workbench outbox for write_note)
    require_approval: bool = True  # external effects always require approval; setting this false for one is a validation error
    output_field: str = "tool_result"
    append_to: str = ""  # optional list field receiving {tool,args,result,callId} for the prompt's tool_results item


@register
class ToolCall(AgentOp):
    type = "agent.tool_call"
    Config = ToolCallConfig

    def reads(self, cfg):
        return [v for t in cfg.args.values() for v in template_vars(t)]

    def writes(self, cfg):
        return [cfg.output_field] + ([cfg.append_to] if cfg.append_to else [])

    def templates(self, cfg):
        return [(f"args/{k}", t) for k, t in cfg.args.items()]

    def effects(self, cfg):
        return list(toolmod.TOOLS[cfg.tool].effects) if cfg.tool in toolmod.TOOLS else []

    def check(self, cfg, spec):
        t = toolmod.TOOLS.get(cfg.tool)
        out = []
        if t is None:
            return [("E_TOOL_UNKNOWN", f"tool '{cfg.tool}' is not registered")]
        if set(cfg.args) != set(t.args):
            out.append(("E_TOOL_ARGS", f"tool '{cfg.tool}' takes arguments {sorted(t.args)} but the block binds {sorted(cfg.args)}"))
        if t.external and not cfg.require_approval:
            out.append(("E_TOOL_APPROVAL", f"'{cfg.tool}' has external effects {list(t.effects)}; an explicit approval interrupt is required and cannot be switched off"))
        if t.name in ("read_text_file",) and not cfg.allowed_dir:
            out.append(("E_TOOL_BOUNDS", "a file-reading tool must declare the directory it is allowed to read"))
        return out

    def execute(self, rt, ctx, cfg, state):
        from langgraph.types import interrupt

        from .memory import args_hash, new_id

        tool = toolmod.TOOLS[cfg.tool]
        args = {k: render_template(t, state) for k, t in cfg.args.items()}
        base = rt.resolve_dir(cfg.allowed_dir) if cfg.allowed_dir else rt.outbox
        call_id = new_id("tool")
        approval = None
        if tool.external:
            # Everything before interrupt() runs again when the node replays after a resume, so this node does nothing effectful above this line.
            decision = interrupt({"type": "approval", "node": ctx.node_id, "tool": cfg.tool, "args": args, "effects": list(tool.effects), "actions": ["approve", "reject", "edit"],
                                  "prompt": f"Approve the external effect {list(tool.effects)} of tool '{cfg.tool}'?", "boundedTo": str(base)})
            approval = decision if isinstance(decision, dict) else {"action": str(decision)}
            rt.emit("interrupt_resumed", ctx.node_id, value=approval, kind="approval")
            if approval.get("action") == "reject":
                res = {"tool": cfg.tool, "args": args, "status": "rejected", "result": None, "effects": list(tool.effects), "approval": approval}
                rt.emit("tool_call", ctx.node_id, callId=call_id, **res)
                return self._out(cfg, res, call_id)
            if approval.get("action") == "edit" and isinstance(approval.get("value"), dict):
                args = {k: str(v) for k, v in approval["value"].items() if k in tool.args}
        status, result, replay = "ok", None, False
        try:
            if tool.external:
                key = rt.effect_key(ctx, cfg.tool, args)
                claim, prior = rt.memory.claim_effect(key, run_id=rt.run_id, thread_id=rt.thread_id, node=ctx.node_id, tool=cfg.tool, args_hash=args_hash(args))
                if claim == "claimed":
                    result = tool.run(args, base)
                    rt.memory.complete_effect(key, result)
                elif claim == "done":
                    result, replay = prior, True
                else:
                    status, replay = "effect_uncertain", True
                    result = {"note": "an earlier attempt started this effect but never recorded completion; it was not repeated automatically"}
            else:
                result = tool.run(args, base)
        except toolmod.ToolError as e:
            status, result = "error", {"code": e.code, "message": e.message}
        res = {"tool": cfg.tool, "args": args, "status": status, "result": result, "effects": list(tool.effects), "approval": approval, "replaySkipped": replay}
        rt.count_tool()
        rt.emit("tool_call", ctx.node_id, callId=call_id, **res)
        return self._out(cfg, res, call_id)

    def _out(self, cfg, res, call_id):
        out = {cfg.output_field: {**res, "callId": call_id}}
        if cfg.append_to:
            out[cfg.append_to] = [{"tool": res["tool"], "args": res["args"], "result": res["result"], "callId": call_id}]
        return out

    def explain(self, cfg, inputs=None, outputs=None):
        t = toolmod.TOOLS.get(cfg.tool)
        return {"summary": (t.description if t else "unknown tool") + (" External effects require an approval interrupt and run once (effect ledger)." if t and t.external else ""),
                "reads": self.reads(cfg), "writes": self.writes(cfg), "effects": self.effects(cfg), "tool": t.describe() if t else None}


# ================================================================================================ human in the loop
class HumanInterruptConfig(Strict):
    prompt: str = "Please review."
    show_field: str = ""  # state field presented to the person (and editable when edit_field is set)
    edit_field: str = ""  # on action 'edit' the supplied value replaces this field
    decision_field: str = "decision"
    actions: list[Literal["approve", "reject", "edit"]] = ["approve", "reject"]


@register
class HumanInterrupt(AgentOp):
    type = "agent.human_interrupt"
    Config = HumanInterruptConfig

    def reads(self, cfg):
        return ([cfg.show_field] if cfg.show_field else []) + template_vars(cfg.prompt)

    def writes(self, cfg):
        return [cfg.decision_field] + ([cfg.edit_field] if cfg.edit_field else [])

    def templates(self, cfg):
        return [("prompt", cfg.prompt)]

    def check(self, cfg, spec):
        return [("E_CONFIG", "action 'edit' needs an edit_field")] if "edit" in cfg.actions and not cfg.edit_field else []

    def execute(self, rt, ctx, cfg, state):
        from langgraph.types import interrupt

        payload = {"type": "human_input", "node": ctx.node_id, "prompt": render_template(cfg.prompt, state), "actions": list(cfg.actions),
                   "proposed": state.get(cfg.show_field) if cfg.show_field else None, "editField": cfg.edit_field or None}
        v = interrupt(payload)  # replay-safe: nothing above has an effect
        v = v if isinstance(v, dict) else {"action": str(v)}
        rt.emit("interrupt_resumed", ctx.node_id, value=v, kind="human_input")
        out: dict[str, Any] = {cfg.decision_field: v.get("action", "approve")}
        if v.get("action") == "edit" and cfg.edit_field:
            out[cfg.edit_field] = v.get("value")
        return out

    def explain(self, cfg, inputs=None, outputs=None):
        return {"summary": "Pauses the thread with a pending-input form; the run resumes from its checkpoint with the person's decision.", "reads": self.reads(cfg), "writes": self.writes(cfg), "effects": []}


# ================================================================================================ memory blocks
class MemorySelectConfig(Strict):
    policy: str = ""
    output_field: str = "memory"
    short_term_field: str = ""  # messages field holding this thread's conversation (the short-term store)


@register
class MemorySelect(AgentOp):
    type = "agent.memory_select"
    Config = MemorySelectConfig

    def reads(self, cfg):
        return ([cfg.short_term_field] if cfg.short_term_field else [])

    def writes(self, cfg):
        return [cfg.output_field]

    def effects(self, cfg):
        return ["memory_read"]

    def check(self, cfg, spec):
        return [("E_POLICY_UNKNOWN", f"memory policy '{cfg.policy}' is not declared in this graph")] if spec.policy(cfg.policy) is None else []

    def execute(self, rt, ctx, cfg, state):
        pol = Policy.model_validate(rt.spec.policy(cfg.policy).model_dump())
        lt = rt.memory.list_records()
        st = short_term_records(state.get(cfg.short_term_field) or [], rt.thread_id) if cfg.short_term_field else []
        from .memory import new_id

        aid = new_id("app")

        def summarize(old, max_chars):
            stage = next(s for s in pol.stages if s.op == "summarize")
            spec = ModelSpec.model_validate(stage.config.get("model") or {"provider": "ollama"})
            text = "\n".join(f"- {r['text']}" for r in old)
            msgs = [{"role": "system", "content": f"Summarize the notes in at most {max_chars} characters. Keep numbers and constraints.", "segments": []},
                    {"role": "user", "content": text, "segments": [{"start": 0, "end": len(text), "source": {"kind": "memory_records_to_summarize", "ids": [r["id"] for r in old]}}]}]
            return rt.model_call(ctx, spec, msgs, purpose="memory_summary")["text"][:max_chars]

        read_state = {k: v for k, v in state.items() if k not in (cfg.output_field,)}
        app = run_policy(pol, lt, st, read_state, summarizer=summarize, meta={"id": aid, "runId": rt.run_id, "threadId": rt.thread_id, "node": ctx.node_id,
                                                                                    "outputField": cfg.output_field, "inputs": _policy_inputs(pol, state, cfg)})
        rt.memory.save_application(app)
        recs = selection_records(app, aid)
        rt.emit("memory_selection", ctx.node_id, applicationId=aid, policy=pol.id, universe=len(app["records"]), selected=len(recs), tokensEstimate=app["tokensEstimate"],
                stages=[{"id": s["id"], "op": s["op"], "in": s["in"], "out": s["out"], "note": s["note"]} for s in app["stages"]],
                excluded=sum(1 for r in app["records"].values() if r["status"] in ("excluded", "summarized")))
        return {cfg.output_field: recs}

    def explain(self, cfg, inputs=None, outputs=None):
        return {"summary": "Applies a visual memory-selection policy (retrieve, filter, rank, deduplicate, token budget, summarize) and records every record's decision.",
                "reads": self.reads(cfg), "writes": self.writes(cfg), "effects": ["memory_read"]}


def _policy_inputs(pol: Policy, state: dict[str, Any], cfg: MemorySelectConfig) -> dict[str, Any]:
    names = set(_vars_root(pol.query))
    for s in pol.stages:
        for sc in s.config.get("scopes", []) or []:
            names |= set(_vars_root(sc))
    return {n: state.get(n) for n in names if n in state}


class MemoryWriteConfig(Strict):
    target: Literal["long_term", "short_term"] = "long_term"
    text: str = "{answer}"
    role: Literal["user", "assistant", "system"] = "assistant"  # short_term
    field: str = "history"  # short_term: the messages field to append to (the thread's conversation)
    namespace: str = "default"
    scope: str = "global"  # template, e.g. user:{user_id}
    kind: Literal["semantic", "episodic", "procedural", "summary"] = "semantic"
    importance: float = Field(0.5, ge=0, le=1)
    metadata: dict[str, str] = {}  # key -> template
    generated: bool = True  # a model-generated assertion is marked generated; storing it does not make it true
    evidence: str = ""  # template naming the source of the fact
    mode: Literal["direct", "approve"] = "direct"  # approve: a visible decision stage (interrupt) accepts or rejects the proposal
    skip_duplicates: bool = True
    min_chars: int = Field(1, ge=0)
    max_chars: int = Field(1000, ge=1)


@register
class MemoryWrite(AgentOp):
    type = "agent.memory_write"
    Config = MemoryWriteConfig

    def reads(self, cfg):
        return template_vars(cfg.text) + template_vars(cfg.scope) + template_vars(cfg.evidence) + [v for t in cfg.metadata.values() for v in template_vars(t)]

    def writes(self, cfg):
        return [cfg.field] if cfg.target == "short_term" else []

    def templates(self, cfg):
        return [("text", cfg.text), ("scope", cfg.scope), ("evidence", cfg.evidence)] + [(f"metadata/{k}", t) for k, t in cfg.metadata.items()]

    def effects(self, cfg):
        return ["memory_write"] if cfg.target == "long_term" else []

    def check(self, cfg, spec):
        out = []
        if cfg.target == "short_term":
            f = spec.field(cfg.field)
            if f is None:
                out.append(("E_STATE_FIELD_UNKNOWN", f"short-term field '{cfg.field}' is not in the state schema"))
            elif f.type not in ("messages", "records"):
                out.append(("E_STATE_TYPE", f"short-term field '{cfg.field}' must be a messages field"))
            elif f.reducer.kind == "replace":
                out.append(("E_REDUCER", f"short-term field '{cfg.field}' needs an append-style reducer to accumulate the conversation"))
        return out

    def execute(self, rt, ctx, cfg, state):
        from langgraph.types import interrupt

        text = render_template(cfg.text, state).strip()
        scope = render_template(cfg.scope, state)
        evidence = render_template(cfg.evidence, state) if cfg.evidence else None
        val = {"ok": True, "errors": []}
        if len(text) < cfg.min_chars:
            val = {"ok": False, "errors": [f"text shorter than min_chars={cfg.min_chars}"]}
        if len(text) > cfg.max_chars:
            val = {"ok": False, "errors": [f"text longer than max_chars={cfg.max_chars}"]}
        if cfg.mode == "approve":
            d = interrupt({"type": "memory_write_proposal", "node": ctx.node_id, "proposal": {"text": text, "namespace": cfg.namespace, "scope": scope, "kind": cfg.kind, "generated": cfg.generated,
                                                                                            "evidence": evidence}, "actions": ["approve", "reject", "edit"],
                           "prompt": "Accept this proposed memory record? A generated assertion is stored as generated; storing it does not establish its truth."})
            d = d if isinstance(d, dict) else {"action": str(d)}
            rt.emit("interrupt_resumed", ctx.node_id, value=d, kind="memory_write_proposal")
            if d.get("action") == "reject":
                rt.emit("memory_write", ctx.node_id, status="rejected", text=text, scope=scope, validation=val)
                return {}
            if d.get("action") == "edit" and isinstance(d.get("value"), str):
                text = d["value"].strip()
        if cfg.target == "short_term":
            msg = {"id": f"msg:{rt.thread_id}:{ctx.node_id}:{ctx.step}", "role": cfg.role, "content": text, "text": text, "ts": time.time(),
                   "source": {"kind": "memory_write", "node": ctx.node_id, "runId": rt.run_id}}
            rt.emit("memory_write", ctx.node_id, status="written", store="short_term", field=cfg.field, text=text, scope=f"thread:{rt.thread_id}", validation=val, recordId=msg["id"])
            return {cfg.field: [msg]}
        if not val["ok"]:
            rt.emit("memory_write", ctx.node_id, status="invalid", store="long_term", text=text, scope=scope, validation=val)
            return {}
        dup = None
        if cfg.skip_duplicates:
            norm = " ".join(text.lower().split())
            dup = next((r for r in rt.memory.list_records(namespace=cfg.namespace, scope=scope) if " ".join(r["text"].lower().split()) == norm), None)
        if dup:
            rt.emit("memory_write", ctx.node_id, status="skipped_duplicate", store="long_term", text=text, scope=scope, validation={**val, "duplicateOf": dup["id"]}, recordId=dup["id"])
            return {}
        from .memory import args_hash

        key = rt.effect_key(ctx, "memory_write", {"text": text, "scope": scope, "ns": cfg.namespace})
        claim, prior = rt.memory.claim_effect(key, run_id=rt.run_id, thread_id=rt.thread_id, node=ctx.node_id, tool="memory_write", args_hash=args_hash({"text": text}))
        if claim != "claimed":
            rt.emit("memory_write", ctx.node_id, status="replay_skipped", store="long_term", text=text, scope=scope, recordId=(prior or {}).get("id") if prior else None)
            return {}
        rec = rt.memory.put_record({"namespace": cfg.namespace, "scope": scope, "kind": cfg.kind, "text": text, "importance": cfg.importance, "generated": cfg.generated,
                                    "metadata": {k: render_template(t, state) for k, t in cfg.metadata.items()}, "source": {"node": ctx.node_id, "runId": rt.run_id, "threadId": rt.thread_id}},
                                   run_id=rt.run_id, thread_id=rt.thread_id, node=ctx.node_id, evidence=evidence, validation=val)
        rt.memory.complete_effect(key, {"id": rec["id"]})
        rt.emit("memory_write", ctx.node_id, status="written", store="long_term", recordId=rec["id"], text=text, scope=scope, namespace=cfg.namespace, kind=cfg.kind, generated=cfg.generated,
                evidence=evidence, validation=val, version=rec["version"])
        return {}

    def explain(self, cfg, inputs=None, outputs=None):
        return {"summary": "Writes a record to the thread's short-term conversation or to the long-term store, with validation, scope, evidence and an audit entry; optionally behind an accept/reject decision.",
                "reads": self.reads(cfg), "writes": self.writes(cfg), "effects": self.effects(cfg)}


def all_agent_ops() -> list[AgentOp]:
    from graph_core.registry import all_ops

    return [o for o in all_ops() if isinstance(o, AgentOp)]


__all__ = ["AgentOp", "NodeCtx", "NodeFailure", "render_prompt", "json_schema_of", "validate_structured", "reducer_fn", "pred_fields"]
