"""OpenAI-compatible chat completions as a LangChain chat model, over httpx (no extra dependency).

Speaks `POST {base_url}/chat/completions` (non-streamed and `stream: true` server-sent events). Works with any server that
implements that protocol: Ollama's `/v1`, llama.cpp, vLLM, LM Studio, or hosted APIs given a key. Usage is the provider's
`usage` object when present (streams ask for it with `stream_options.include_usage`); nothing is estimated here.
"""
from __future__ import annotations

import json
from typing import Any, Iterator
from urllib.parse import urlparse

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import SecretStr

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def check_base_url(url: str | None, has_key: bool) -> str:
    """http(s) only; a key is never sent over plain http to another host."""
    if not url:
        raise ValueError("The OpenAI-compatible provider needs a base_url such as http://127.0.0.1:11434/v1.")
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ValueError("base_url must be an http(s) URL.")
    if has_key and p.scheme != "https" and p.hostname not in LOOPBACK:
        raise ValueError("An API key is only sent over https (or to this machine).")
    return url.rstrip("/")


def is_loopback(url: str | None) -> bool:
    return bool(url) and urlparse(url).hostname in LOOPBACK


def _role(m: BaseMessage) -> str:
    return {"system": "system", "ai": "assistant", "human": "user"}.get(m.type, "user")


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content or [])


def _usage(u: dict | None) -> dict | None:
    if not u or (u.get("prompt_tokens") is None and u.get("completion_tokens") is None):
        return None
    i, o = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
    return {"input_tokens": i, "output_tokens": o, "total_tokens": u.get("total_tokens") or i + o}


class OpenAICompatibleChat(BaseChatModel):
    model: str
    base_url: str
    api_key: SecretStr | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    timeout: float = 180.0
    reasoning_effort: str | None = None

    @property
    def _llm_type(self) -> str:
        return "openai-compatible"

    def _body(self, messages: list[BaseMessage], stream: bool) -> dict:
        body: dict[str, Any] = {"model": self.model, "messages": [{"role": _role(m), "content": _text(m.content)} for m in messages], "stream": stream}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        if self.seed is not None:
            body["seed"] = self.seed
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    def _client(self) -> httpx.Client:
        headers = {"Authorization": f"Bearer {self.api_key.get_secret_value()}"} if self.api_key else {}
        return httpx.Client(base_url=self.base_url, headers=headers, timeout=self.timeout, trust_env=False)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        with self._client() as c:
            r = c.post("/chat/completions", json=self._body(messages, False))
            r.raise_for_status()
            data = r.json()
        choice = data["choices"][0]
        msg = AIMessage(content=choice["message"].get("content") or "", usage_metadata=_usage(data.get("usage")),
                        response_metadata={"model_name": data.get("model"), "stop_reason": choice.get("finish_reason")})
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]:
        with self._client() as c, c.stream("POST", "/chat/completions", json=self._body(messages, True)) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                usage = _usage(data.get("usage"))
                for choice in data.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content") or ""
                    meta = {"model_name": data.get("model"), "stop_reason": choice.get("finish_reason")} if choice.get("finish_reason") else {}
                    if delta or meta:
                        yield ChatGenerationChunk(message=AIMessageChunk(content=delta, response_metadata=meta))
                if usage:  # sent in a final chunk with empty choices
                    yield ChatGenerationChunk(message=AIMessageChunk(content="", usage_metadata=usage))
