# ADR0062: bounded pure calculator turns in releases

Status: accepted; native/live/editor/recovery verified on Mac arm64, 2026-10-06.

HANDOFF§59 item 1 asks for effect-free tools in serving. Add `agent_tools`, node
`__agent_tools_graph__`, in a new `production/tools_agent_adapter.py`; preserve the
existing agent/conversation/retrieval adapter bytes and native compiler/blocks.

- Accept stateless prompt/set_state/calculator/retrieve/at most one local Ollama
  chat model. Require at least one calculator. Optional 0–2 indexes use ADR0060's
  exact pinned snapshot/provider mechanism; serving never reads the source corpus.
  File-read/file-write tools, interrupts, memory policies and thread state refuse.
- Require maxToolCalls 1–8 in addition to ADR0023's graph/state/input/model/step/time
  limits. The runtime reserves a tool call atomically before native execution,
  including parallel nodes; native counting does not count the reservation twice.
  Exhaustion refuses the entire result, never returning a partial success.
- Calculator templates may read only fields no node writes. Render with defaults
  and the request during validation; bound to 512 characters, 128 AST nodes,
  depth 32, finite literals/results of magnitude at most 1e100; exponentiation
  refuses. Evaluate through the unchanged native calculator. Its ordinary argument
  errors remain native tool error data. These additional serving limits avoid
  arbitrarily expensive powers without changing research-tool identities.
- Fresh native saver/private temporary workbench each request. Registration pins
  completed END graph/config/final-state membership, reference input, environment,
  tool source and own/reused adapter files; verify before/after execution. No source
  thread changes. Native call count always recorded; tool arguments/results and
  full state/events/context follow captureInputs. Generated output can echo input.
- Shared release, admission, deadlines, cancellation, replay, SSE and idempotency
  apply. Candidate/registry/request controls include this family. Monitoring reads
  recorded warmup/traces; independently supplied text labels give literal agreement.
  Integration also repairs ADR0060 retrieval text labels and monitoring dispatch.
- Compatibility: old plain agents, conversations, JSON conversations, retrieval and
  non-agent families retain their execution source hashes. Shared runtime/API/monitor
  integration files are pinned by stateless `agent_json` (ADR0045), so existing
  `agent_json` versions require re-registration. No weakened identity check, native
  dependency, schema or training change. An explicit baseline hash fixture verifies
  old execution sources against commit 6f63e09 in shallow hosted checkouts too.

Verification is recorded in HANDOFF§60: focused native HTTP/replay/privacy/parallel
budget/source-identity tests; actual Ollama context/usage; owned Chrome source →
register → warmup/deploy → request/replay → independent label/monitor; complete
native/live/editor suites and source-deletion recovery. Hosted acceptance is recorded separately from local evidence. No semantic model-quality, GPU, external file snapshot, effect approval,
model-chosen tools, conversation tools or distributed execution claim.
