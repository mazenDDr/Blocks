# ADR0053: streamed agent serving responses

Status: accepted on Mac arm64 with real local Ollama (2026-10-06); hosted verification recorded separately in HANDOFF§50.

Agent, conversation and JSON-agent releases (ADR0023–0024, ADR0045) answer only
after the whole native turn finishes. Clients that show generated text had to
wait for the full response.

Decision: a streaming variant of the existing predict route,
`POST /api/serve/{target}/{namespace}/predict/stream`, with server-sent events.

- The request goes through exactly the same `ProductionRuntime.predict` path:
  routing, idempotency, admission, deadlines, cancellation, sessions, budgets,
  trace persistence. It runs in a worker thread whose context carries a token sink.
- `invoke_chat` streams through LangChain only when a sink is set in the current
  context; otherwise it is unchanged. Streamed chunks are added together into the
  same message object `invoke` returns, so recorded text, provider usage and
  response metadata keep their meaning. (Checked against local Ollama: identical
  text, usage and metadata for the same seeded request.)
- Events: `token` with each provider text delta in arrival order, then `result`
  with the exact recorded trace the plain route returns, then `end` with delta
  count, characters and status. Errors raised before a trace exists end with an
  `error` event. Non-finite inputs are refused before streaming, as on the plain
  route. Authentication is the same middleware.

The recorded trace stays authoritative. Deltas are provisional: a turn whose
output is later refused (deadline, cancellation, budget, JSON validation) has
already streamed the text it received, and a multi-call turn streams every call's
deltas in order. A client disconnect does not cancel the request; the existing
cancel route does. The Production Requests tab has a "Stream prediction request"
button for agent releases that shows the provisional text as it arrives and then the
recorded trace (`src/sse.ts` parses frames). No token-level timing record, no streaming for
non-agent adapters (they end with `result` and no tokens), no Anthropic streaming
evidence (no credentials).

Verification: 3 offline tests (unchanged behaviour without a sink, deltas
concatenate to the returned text, sink scoped to its context across threads) and
2 live tests against real local Ollama qwen3.5:2b on a real native source run and
release: streamed deltas equal the recorded `response` exactly, the final event
equals the stored trace, the plain route then replays it idempotently, and
refusals (non-finite input, unknown target) keep their behaviour. 2 Node tests
cover the SSE parser, and the real-Ollama JSON agent browser journey streams a
request from the editor: the shown text equals that request's recorded response.
