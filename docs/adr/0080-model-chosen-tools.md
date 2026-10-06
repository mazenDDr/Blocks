# ADR0080: model-chosen tools

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded in HANDOFF§87.

Tools ran only as explicit graph nodes with templated arguments: "a model does not
choose tools". Models served by Ollama and OpenAI-compatible servers can request tool
calls, and nothing used that.

Decision: block `agent.tool_agent`.

- Offers declared tools to the model as function schemas and runs the calls it
  requests through the existing tool implementations, feeding each result back
  (`role: tool`) until the model answers. At most `max_tool_calls` (1–8) calls; once
  spent, the model is asked again with no tools offered and must answer. Graph limits
  (`maxModelCalls`, `maxToolCalls`) still apply.
- Only effect-free tools may be offered: `calculator` and `read_text_file` (bounded to
  a declared directory). `write_note` and any tool with external effects stay behind
  the explicit tool node with its approval interrupt; a model cannot trigger them.
- Requests for a tool that was not offered, extra calls past the limit and wrong
  argument names are refused without running anything, recorded, and returned to the
  model as tool results.
- Every call is recorded (`tool_call` events with `chosenBy: "model"`, the model call
  that requested it, arguments, status and result; `model_call` events list the
  offered tools and requested calls) and optionally written to a list field.
- Providers: `ollama` (native tool calling) and `openai_compatible` (`tools`,
  `tool_calls` and `role: tool` messages on the wire). Tool turns are not streamed.
- Editor: form (model, fields, offered tools, readable directory, call limit) and card
  summary; example `tool_agent_calculator`.

Observed (local Ollama, temperature 0, seed 7): asked to use the calculator for
1234 × 5678, qwen3.5:2b without thinking answered 5,909,752 without calling the tool;
with thinking it called the tool. qwen3.5:4b and gemma4:12b called it either way;
qwen3.5:4b then answered 7,006,652 via both native and `/v1`. Whether a model uses a
tool is the model's behaviour, which the recorded events make visible.

Not provided: model-chosen tools in served releases, external-effect tools for models,
parallel tool execution, tool calling with the Anthropic or fixture providers, user-defined
tools.

Compatibility: `agent/blocks.py`, `agent/models.py` and `agent/runtime.py` are pinned by
agent serving families; versions registered before this change must be re-registered
(`tests/fixtures/serving_sources_adr0080.json`).

Verification: `tests/test_tool_agent.py` (offline against a scripted local
OpenAI-compatible server: tool offered with its schema, call executed, result sent back
as a tool message, answer recorded; unoffered tool refused without running; the budget
forces a tool-free final request; validation refusals; live: qwen3.5:4b calls the
calculator and answers correctly through native Ollama and `/v1`); browser journey
`tools/editor_tool_agent_block_smoke.py`.
