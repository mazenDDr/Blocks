# ADR0082: serving model-chosen calculator turns

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded in HANDOFF§90.

ADR0080 let a model choose tools in research runs only; no release could serve it.

Decision: serving family `agent_tool_choice` (`production/tool_choice_adapter.py`,
candidate node `__agent_tool_choice_graph__`).

- Graph contract: prompt / set_state nodes and exactly one `agent.tool_agent` offering
  only the effect-free calculator (no file tools, no directory), `max_tool_calls` ≤ 4;
  stateless (no thread-scoped fields), no memory policies or indexes; pinned local
  Ollama with thinking off, `max_tokens` ≤ 256, `timeout_s` ≤ 60; limits declared with
  `max_tool_calls + 1 ≤ maxModelCalls ≤ 5`, `maxToolCalls` ≤ 4, `maxSeconds` ≤ 120,
  `maxTokens` ≤ 8192.
- Each request runs the native graph in an isolated temporary store. Every tool call
  the model requested (tool, arguments, status, result, the model call that asked) is
  returned in the request evidence, so it is recorded in the trace even when input
  capture is off; full state, events and model contexts only with capture on.
- Identity pins this adapter, the tool implementations and the OpenAI-compatible module
  with the usual agent files; the model digest is pinned at registration.

Not provided: file-reading or external-effect tools in releases, model choice over the
OpenAI-compatible provider in releases, conversations with tools, streaming tool turns.

Compatibility: `production/runtime.py` and `services/control/production_api.py` changed
(agent family list, candidate listing); agent JSON-family versions pinning them must be
re-registered, as after ADR0075.

Verification: `tests/test_production_tool_choice.py` (example satisfies the contract;
six refusals with their codes; live: source run → register → release → a served
request where qwen3.5:4b called the calculator for 987 × 654 and answered 645498, with
the call recorded in the stored trace while capture was off).
