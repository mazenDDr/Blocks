# ADR0059: native JSON conversations

Status: accepted on Mac arm64 with real local Ollama (2026-10-06); hosted verification recorded separately in HANDOFF§56.

Two serving families existed for agent graphs that people talk to over several
turns or that must return structured data: native conversations (ADR0024: thread
state restored from per-release/user/session checkpoints, text output) and JSON
turns (ADR0045: one schema-validated object, stateless). A graph that keeps
conversation state and answers each turn with a validated object could not be
served.

Decision: a third adapter, `conversation_json`, registered with node
`__agent_json_conversation__`, in its own module `production/json_conversation_adapter.py`.
Both existing adapters pin their own source files in registered identities, so the
new module reuses their functions without editing them.

- Contract: prompt/set_state nodes plus exactly one structured-output node; no
  tools, retrieval, memory or interrupts; at least one thread-scoped state field;
  1–4 turn-scoped text inputs; the served object is a separate turn-scoped object
  field with a replace reducer written only by the structured-output node;
  `on_failure=fail` with retries inside the model-call budget; 1–12 flat schema
  fields; the JSON adapter's provider, limit and template bounds.
- A turn restores the last successful native checkpoint for that release, user
  and session, runs the graph, validates the object against the native schema
  and only then produces a checkpoint candidate. Commit, refusal on deadline,
  cancellation and budget, session serialization, reset/fork/restore/history and
  discovery are the existing conversation machinery; label quality is the JSON
  adapter's canonical JSON agreement; releases must use `sessionMode=conversation`.
- Identity pins the conversation adapter's files plus the JSON adapter and this
  module, but not the HTTP layer, runtime or monitor, so routine API changes do
  not invalidate these versions.
- Candidates, registration, release checks, prediction (plain and streamed),
  replay with the conversation parent, labels, monitoring and the editor's
  Production workspace treat it like the other agent families.

Not provided: tools, retrieval, memory or interrupts in served graphs, Anthropic or
other providers, nested schemas, or semantic quality benchmarks.

Verification: 3 offline contract tests (acceptance and six refusals, identity file
scope) and 2 live tests with real local Ollama qwen3.5:2b: a real source run,
registration as `conversation_json`, a conversation release, two turns returning
exactly the requested objects on the same native thread with `turns`/`history`
advanced in the checkpoint, an independent second session, canonical JSON labels
accepted and schema-invalid labels refused, the source thread unchanged, and
refusals for stateless releases and stateless graphs.
