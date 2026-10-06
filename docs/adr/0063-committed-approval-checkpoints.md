# ADR0063: committed native approval checkpoints in releases

Status: accepted; Mac native/live/editor/recovery verified, 2026-10-06.

HANDOFF§59 item2 requires a pending served turn and a reviewed follow-up decision.
Add `conversation_approval` / `__agent_approval_conversation__` using new modules
`approval_adapter`, `approval_store`, `approval_requests`, `approval_monitor`, and
`control/approval_api`; existing END-only conversation adapter/store bytes stay intact.

- Accept an acyclic native prompt/set_state/at most one local Ollama chat graph with
  exactly one `human_interrupt`, one active path (conditional routes, no fixed
  parallel forks/joins), thread state, immutable 1–4 text inputs, and replace
  text decision/edit fields. Reuse native compiler/blocks/interrupt/Command/saver.
  The ADR0024 provider, graph/state/context/checkpoint limits apply. File tools,
  retrieval, memory effects, JSON outputs, empty/duplicate actions and repeated
  interrupts refuse.
- Registration requires a completed END research run (actually reviewed in research).
  Warmup executes privately until its native interrupt; it creates no live head and
  fabricates no decision or final prediction. Returned HTTP202 has no predictions,
  the exact native pending review, and a committed paused checkpoint.
- Serialize native checkpoint + metadata + pending task writes through native serde.
  Resume restores both checkpoint and pending writes into a fresh saver, then uses
  `Command(resume={interruptId: decision})`. Earlier completed nodes/model calls
  are not repeated. One acyclic interrupt bounds a logical turn to two segments.
  Model/token budgets and active execution seconds carry across the pause; human
  waiting time is excluded. New END-to-next-turn execution resets turn budgets.
- Explicit session required. Inspection is read-only and returns current head,
  pending review, persistent state/input and budget. Resume requires expected release,
  revision, checkpoint SHA and interrupt ID, with approve/reject or text edit <=2000
  characters. It preserves paused inputs and refuses undeclared/stale decisions.
  Reject writes the native decision field; the registered graph defines its branch.
- Reuse admission, deadlines, cancellation, per-session serialization, idempotency
  and optimistic SQLite head checks. New transaction code commits paused/END head
  and terminal request trace atomically; cancelled/late/failed candidates do not
  advance state. A committed pending HTTP request is terminal; its session remains
  paused across server restart. A fresh competing decision loses the reviewed head.
- Native state, review payload and approval receipt persist even with captureInputs
  off. Raw native events/model context still follow trace capture. Read-only monitoring
  counts committed paused/END segments without duplicating model calls; pending is
  distinguished from failure. Warmup has no output reference for prediction drift.
  Isolated replay can reproduce a captured paused turn or reviewed resume; no head
  changes. Reset/fork/historical restore refuse paused heads; END actions keep their
  original machinery. Editor adds a dedicated reviewed approval component.
- No new dependency/schema change. Plain agent/conversation/JSON-conversation/
  retrieval/calculator/non-agent execution source hashes stay unchanged. Stateless
  `agent_json` pins shared runtime/API/monitor, so those versions need re-registration.
  The new approval family pins its own behavior and request/HTTP/monitor modules.

Verification and retained failures are in HANDOFF§61. Native tests cover approve,
reject, edit, optimistic conflict/concurrency, pending-write rehydration, capture-off
persistence, cancellation, deadlines, failed SQLite atomicity, independent native-state
agreement, actual SIGKILL restart, strict limits and scope refusals. Actual installed
Ollama verifies the pre-interrupt context/call survives resume without a second call.
Owned Chrome verifies source review → registration → paused warmup → deploy →
pending inspection → edit/approve/reject, and sealed source-deletion recovery of a
still-paused checkpoint. Full native 1319 passed / 1 skipped / 21 deselected; live21 passed; all19 editor
journeys and sealed recovery passed. Hosted acceptance is recorded separately.

This establishes approval checkpoint machinery, not external effects. `write_note`
requires a durable effect ledger and a separately bounded/pinned output contract;
no external file write, distributed ownership or exactly-once effect claim is made.

Hosted calculator CI exposed a selection timeout in the existing layout-groups
journey after reload (empty runtime/API/console arrays). Its actual pointer click
now waits for stable frame geometry after opening the arrangement panel, and
clicks within the header. Exact three-card selection/geometry/Undo assertions remain.
The observed failure and repair evidence are recorded separately in HANDOFF§61;
layout timing is an inference until hosted verification of this fix.
