# ADR0076: OpenAI-compatible chat provider

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded in HANDOFF§83.

Agent graphs could call local Ollama (native API) or Anthropic (needs a key). Many
model servers speak the OpenAI chat-completions protocol (llama.cpp, vLLM, LM Studio,
Ollama's own `/v1`, hosted APIs), and no provider reached them.

Decision: provider `openai_compatible` for research runs.

- `agent/openai_compat.py`: a LangChain chat model over httpx (already a dependency;
  no `langchain-openai`). `POST {base_url}/chat/completions`, non-streamed and
  `stream: true` server-sent events (`stream_options.include_usage` for usage).
  Usage is the server's `usage` object when sent; nothing is estimated.
- `base_url` is required and must be http(s). An API key is optional, given only as a
  secret reference (environment variable or secrets file, as for Anthropic), and is
  refused over plain http to any host other than this machine (`E_MODEL_ENDPOINT`, at
  validation and at call time).
- `reasoning_effort` (`none|low|medium|high`) is sent only when set: reasoning models
  otherwise spend `max_tokens` on hidden reasoning (observed with qwen3.5 through
  Ollama `/v1`: empty content, `finish_reason: length`), and some servers reject the
  field.
- Declared effects: `local_runtime` for a loopback base URL, `network` otherwise.
  Cost is recorded as unknown (no pricing table).
- Editor: provider choice, base URL, optional key reference and reasoning effort in
  the model form. While building its journey the agent inspector was found to show
  omitted settings as "(not declared)"/"undefined" although native execution uses the
  op's declared defaults; forms and cards now show the effective defaults.

Serving is unchanged: serving contracts still require pinned local Ollama. Structured
output with this provider requests JSON in the prompt and validates it (no constrained
decoding claimed).

Compatibility: `agent/models.py`, `agent/blocks.py` and `agent/validate.py` are pinned
by agent serving families; versions registered before this change must be
re-registered (`tests/fixtures/serving_sources_adr0076.json`).

Not provided: tool/function calling over the protocol, JSON-schema `response_format`,
serving with this provider, embeddings over the protocol, retries/rate limits, hosted
APIs verified with real keys (no credentials here).

Verification: `tests/test_openai_compatible.py` — offline against a real local HTTP
stub (settings sent, roles, usage, streaming deltas, Authorization only with a key,
reasoning_effort only when set, endpoint rules, effects, validation diagnostic, a
recorded research run) and live against Ollama's real `/v1` (invoke, 11-chunk stream,
research run with qwen3.5:0.8b); browser journey `tools/editor_provider_smoke.py`.
