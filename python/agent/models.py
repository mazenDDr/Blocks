"""Model providers behind LangChain chat-model / embeddings interfaces.

* `ollama`    - local server (langchain-ollama ChatOllama). The one live-tested path.
* `anthropic` - langchain-anthropic ChatAnthropic; the API key comes from a secret REFERENCE (connectors.secrets), never a literal.
* `fixture`   - a deterministic scripted chat model for control-flow tests. Every record made with it is labelled `fixture: true`;
                it is never presented as a real model response.

Provider capabilities differ; only supported generation settings are applied, and the resolved values (plus what was ignored and
why) are returned so they can be recorded with the call (VISION 12.1)."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

OLLAMA_URL = "http://localhost:11434"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"

CAPABILITIES: dict[str, dict[str, Any]] = {
    "ollama": {"label": "Ollama (local runtime)", "settings": ["temperature", "max_tokens", "seed", "timeout_s", "think", "base_url"],
               "structured": "json-schema constrained decoding (Ollama `format`)", "usage": "provider-reported (prompt_eval_count / eval_count)", "live": True},
    "anthropic": {"label": "Anthropic API", "settings": ["temperature", "max_tokens", "timeout_s", "api_key"],
                  "structured": "JSON requested in the prompt, validated here (no constrained decoding is claimed)", "usage": "provider-reported (usage)",
                  "unsupported": {"seed": "the Anthropic API has no sampling seed", "think": "extended thinking is not exposed by this adapter"}, "live": False},
    "fixture": {"label": "FIXTURE (scripted test model, not a language model)", "settings": [], "structured": "scripted JSON replies", "usage": "none (unknown)",
                "unsupported": {"temperature": "scripted", "max_tokens": "scripted", "seed": "scripted"}, "live": False},
}


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["ollama", "anthropic", "fixture"] = "ollama"
    model: str = "qwen3.5:2b"
    temperature: float | None = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(512, ge=1, le=200000)
    seed: int | None = None
    timeout_s: float = Field(180.0, gt=0, le=3600)
    think: bool = False  # ollama reasoning models: request the reasoning trace or not
    base_url: str | None = None
    api_key: dict[str, str] | None = None  # secret reference {"kind":"env","name":...} / {"kind":"file",...}; anthropic only
    # fixture only: rules are checked in order against the full text sent; `responses` is cycled by call number; `fail_first` returns
    # unparsable text for that many calls (to exercise structured-output retries).
    fixture: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _provider_default_model(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("provider") == "anthropic" and "model" not in data:
            return {**data, "model": DEFAULT_ANTHROPIC_MODEL}
        return data


class ModelUnavailable(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def estimate_tokens(text: str) -> int:
    """Labelled ESTIMATE used wherever the provider reports no count: ceil(characters / 4)."""
    return math.ceil(len(text) / 4) if text else 0


def resolve_settings(spec: ModelSpec) -> tuple[dict[str, Any], dict[str, str]]:
    """(resolved generation settings, ignored settings with the reason). Only supported settings are applied."""
    cap = CAPABILITIES[spec.provider]
    resolved: dict[str, Any] = {"model": spec.model}
    ignored: dict[str, str] = {}
    for k in ("temperature", "max_tokens", "seed", "think"):
        v = getattr(spec, k)
        if k in cap["settings"]:
            resolved[k] = v
        elif v not in (None, False) and not (k == "temperature" and v == 0.0 and spec.provider == "fixture"):
            ignored[k] = cap.get("unsupported", {}).get(k, "not supported by this provider")
    resolved["timeout_s"] = spec.timeout_s
    return resolved, ignored


# ------------------------------------------------------------------------------------------------ fixture chat model
def _fixture_model(spec: ModelSpec, counter: dict[str, int]):
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    fx = spec.fixture

    class FixtureChatModel(BaseChatModel):  # type: ignore[misc]
        @property
        def _llm_type(self) -> str:
            return "fixture"

        def _generate(self, messages, stop=None, run_manager=None, **kw):  # noqa: ANN001
            text = "\n".join(str(m.content) for m in messages)
            n = counter.get("n", 0)
            counter["n"] = n + 1
            if n < int(fx.get("fail_first", 0)):
                reply = fx.get("bad_reply", "I cannot produce JSON right now")
            else:
                reply = None
                for r in fx.get("rules", []):
                    if r.get("when_contains") in text and (r.get("when_not_contains") is None or r["when_not_contains"] not in text):
                        reply = r["reply"]
                        break
                if reply is None:
                    resp = fx.get("responses") or []
                    idx = n - int(fx.get("fail_first", 0))
                    reply = resp[min(idx, len(resp) - 1)] if resp and fx.get("clamp", True) else (resp[idx % len(resp)] if resp else fx.get("default", "fixture reply"))
                if not isinstance(reply, str):
                    reply = json.dumps(reply)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

    return FixtureChatModel()


def build_chat_model(spec: ModelSpec, counter: dict[str, int] | None = None):
    """Return a LangChain chat model for `spec`. Raises ModelUnavailable with a stable code when it cannot be built."""
    if spec.provider == "fixture":
        return _fixture_model(spec, counter if counter is not None else {})
    if spec.provider == "ollama":
        from langchain_ollama import ChatOllama

        kw: dict[str, Any] = {"model": spec.model, "base_url": spec.base_url or OLLAMA_URL, "reasoning": bool(spec.think),
                              "client_kwargs": {"timeout": spec.timeout_s}}
        if spec.temperature is not None:
            kw["temperature"] = spec.temperature
        if spec.max_tokens is not None:
            kw["num_predict"] = spec.max_tokens
        if spec.seed is not None:
            kw["seed"] = spec.seed
        return ChatOllama(**kw)
    if spec.provider == "anthropic":
        from connectors import secrets as sec
        from connectors.errors import SourceError

        if not spec.api_key:
            raise ModelUnavailable("E_MODEL_NO_KEY", "The Anthropic provider needs an API key given as a secret reference (environment variable or secrets file).")
        try:
            key = sec.resolve(sec.normalize_ref(spec.api_key, field="api_key"), "api_key")
        except SourceError as e:
            raise ModelUnavailable("E_MODEL_NO_KEY", e.message if hasattr(e, "message") else str(e))
        from langchain_anthropic import ChatAnthropic
        from pydantic import SecretStr

        kw = {"model": spec.model or DEFAULT_ANTHROPIC_MODEL, "api_key": SecretStr(key), "timeout": spec.timeout_s, "max_retries": 0}
        if spec.max_tokens is not None:
            kw["max_tokens"] = spec.max_tokens
        if spec.temperature is not None:
            kw["temperature"] = spec.temperature
        return ChatAnthropic(**kw)
    raise ModelUnavailable("E_MODEL_PROVIDER", f"unknown provider {spec.provider}")


def to_lc_messages(messages: list[dict[str, Any]]):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    out = []
    for m in messages:
        cls = {"system": SystemMessage, "assistant": AIMessage}.get(m["role"], HumanMessage)
        out.append(cls(content=m["content"]))
    return out


def usage_of(msg: Any) -> dict[str, Any]:
    u = getattr(msg, "usage_metadata", None)
    if u and (u.get("input_tokens") is not None or u.get("output_tokens") is not None):
        return {"inputTokens": u.get("input_tokens"), "outputTokens": u.get("output_tokens"), "source": "provider"}
    return {"inputTokens": None, "outputTokens": None, "source": "unavailable"}


def invoke_chat(spec: ModelSpec, messages: list[dict[str, Any]], counter: dict[str, int] | None = None, json_schema: dict | None = None) -> dict[str, Any]:
    """One provider call. Returns text, provider-reported usage (or unavailable), latency. Never fabricates a response."""
    model = build_chat_model(spec, counter)
    if json_schema is not None and spec.provider == "ollama":
        model = model.bind(format=json_schema)
    t0 = time.perf_counter()
    try:
        out = model.invoke(to_lc_messages(messages))
    except ModelUnavailable:
        raise
    except Exception as e:  # noqa: BLE001  (connection refused, model missing, HTTP error ...)
        raise ModelUnavailable("E_MODEL_CALL", f"{spec.provider} call failed: {type(e).__name__}: {str(e)[:300]}")
    latency = (time.perf_counter() - t0) * 1000
    content = out.content
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return {"text": str(content), "usage": usage_of(out), "latencyMs": round(latency, 1), "responseMetadata": {k: v for k, v in (getattr(out, "response_metadata", None) or {}).items()
                                                                                                        if k in ("model", "done_reason", "stop_reason", "total_duration", "eval_duration", "model_name")}}


def ollama_context_length(model: str, base_url: str | None = None) -> int | None:
    """The model's context window when the local runtime reports it (None otherwise; never guessed)."""
    try:
        import httpx

        r = httpx.post((base_url or OLLAMA_URL) + "/api/show", json={"model": model}, timeout=5)
        info = r.json().get("model_info", {})
        for k, v in info.items():
            if k.endswith(".context_length"):
                return int(v)
    except Exception:  # noqa: BLE001
        return None
    return None


def ollama_status(base_url: str | None = None) -> dict[str, Any]:
    try:
        import httpx

        r = httpx.get((base_url or OLLAMA_URL) + "/api/tags", timeout=3)
        return {"reachable": True, "models": sorted(m["name"] for m in r.json().get("models", []))}
    except Exception as e:  # noqa: BLE001
        return {"reachable": False, "models": [], "error": type(e).__name__}


# ------------------------------------------------------------------------------------------------ embeddings
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset("a an and are as at be been by do does for from has have how i in is it its of on or that the their this to was were what when which who why will with you your "
                      "may should must can could would not no than then there these those so if into out up over under about after before".split())


class Embedder:
    """Embeddings behind one interface. `identity` is recorded with every index and retrieval so embedding vectors from different
    models are never mixed. `local_hash` is a deterministic hashed bag-of-words vector: lexical overlap only, NOT a semantic model."""

    def __init__(self, spec):
        self.spec = spec
        if spec.provider == "local_hash":
            self.identity = f"local_hash:d{spec.dimension}:norm{int(spec.normalize)}"
            self.label = "local_hash (deterministic hashed bag-of-words with a small English stopword list; lexical overlap, NOT a semantic model)"
        else:
            self.identity = f"ollama:{spec.model}:norm{int(spec.normalize)}"
            self.label = f"ollama {spec.model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        if self.spec.provider == "local_hash":
            d = self.spec.dimension
            out = np.zeros((len(texts), d), dtype="float32")
            for i, t in enumerate(texts):
                for w in (x for x in TOKEN_RE.findall(t.lower()) if x not in STOPWORDS):
                    h = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "big")
                    out[i, h % d] += 1.0
        else:
            import httpx

            vecs: list[list[float]] = []
            for b in range(0, len(texts), self.spec.batchSize):
                try:
                    r = httpx.post((self.spec.baseUrl or OLLAMA_URL) + "/api/embed", json={"model": self.spec.model, "input": texts[b:b + self.spec.batchSize]}, timeout=120)
                    r.raise_for_status()
                    vecs += r.json()["embeddings"]
                except Exception as e:  # noqa: BLE001
                    raise ModelUnavailable("E_EMBED_CALL", f"ollama embeddings failed: {type(e).__name__}: {str(e)[:200]}")
            out = np.array(vecs, dtype="float32")
        if self.spec.normalize:
            n = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.where(n == 0, 1, n)
        return out.tolist()

    @property
    def dimension(self) -> int:
        return self.spec.dimension if self.spec.provider == "local_hash" else len(self.embed(["x"])[0])


def cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return 0.0 if na == 0 or nb == 0 else sum(x * y for x, y in zip(a, b)) / (na * nb)
