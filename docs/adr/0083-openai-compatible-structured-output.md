# ADR0083: constrained structured output over the OpenAI-compatible provider

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded in HANDOFF§91.

ADR0076 requested JSON from OpenAI-compatible servers only in the prompt. Servers that
implement `response_format: {type: "json_schema"}` (Ollama `/v1`, vLLM, llama.cpp, hosted
APIs) can constrain decoding to the schema.

Decision: when a structured-output block (or any caller passing a JSON schema) uses the
`openai_compatible` provider, the schema is sent as
`response_format: {type: "json_schema", json_schema: {name: "result", strict: true, schema}}`.
The reply is still validated against the schema here, with the block's retry policy, so
servers that ignore the field are handled as before. Not sent when no schema is given.

Compatibility: `agent/models.py` is pinned by agent serving families; declared in
`tests/fixtures/serving_sources_adr0083.json` (re-register agent versions).

Verification: `tests/test_openai_compatible.py` — the request carries `response_format`
exactly when a schema is given (local stub); live, a structured-output block on
qwen3.5:2b through Ollama `/v1` extracted `{"colour": "teal", "count": 3}`.
