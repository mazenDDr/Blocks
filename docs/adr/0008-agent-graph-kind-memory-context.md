# ADR 0008: The `agent` graph kind, memory as a visible subsystem, and per-call context recording

Status: accepted (Milestone 4). Builds on ADR 0001/0002/0003. Sources: VISION 4 ("Different graph semantics must remain explicit"), 12.1-12.8, 13, 17.1-17.3,
23 (Milestone 4), 24 (A14, A15, A30, A31, A32), 26.

## Decisions

1. **A new graph kind, native LangGraph.** `graphKind: "agent"`, `backend: "langgraph"`. `validate(graph)` dispatches on the kind (one entry point, the same diagnostic
   contract). An agent graph compiles to a **native `langgraph.graph.StateGraph`** (`agent/runtime.py:compile_graph`): the state is a `TypedDict` whose reduced fields carry
   their reducer as `Annotated` metadata (so LangGraph builds `BinaryOperatorAggregate` / `LastValue` channels itself), fixed transitions are `add_edge`, routes are
   `add_conditional_edges`, joins are multi-source `add_edge`, pauses are `langgraph.types.interrupt`, resumption is `Command(resume=...)`, step limits are LangGraph's
   `recursion_limit`. A test compares the compiled visual graph with a handwritten native reference. LangChain is used only for what it is good at: chat-model interfaces
   (`langchain-ollama`, `langchain-anthropic`), `langchain-text-splitters`, and `BaseChatModel` for the labelled test fixture. The distinction in VISION 12 (LangChain models/tools,
   LangGraph orchestration) is preserved: nothing re-implements graph execution.
2. **Control edges and routes are not tensor wires.** Edges have kind `control` and ports `in` / `out`; `START` and `END` are reserved ids. A node leaves either by fixed
   transitions or by exactly one **route** (`E_ROUTE_AND_EDGE`). Routes live in the graph's `agent` section next to the typed state schema, joins, limits, indexes and memory
   policies. The `agent` section is part of the semantic hash (case order is semantic); the layout (positions) stays in the UI document.
3. **Typed state with per-field reducers chosen from a small visual set**: `replace`, `append`, `append_unique`, `add`, `max`, `min`, `merge`, `keep_last_n(N)`. Which reducers
   fit which field type is validated (`E_REDUCER`). Two branches that start together and write the same `replace` field are rejected before running (`E_CONCURRENT_WRITE`),
   which is exactly the condition LangGraph rejects at run time. Reduced fields use `Annotated[Any, fn]` so the first write initializes them (a `min` reducer must not start from 0).
   `scope: turn` fields are reset at the start of each run on a thread (`Overwrite` for reduced fields); `scope: thread` fields (a conversation) persist in the checkpoint.
4. **Predicates are data, never code.** A predicate is a tree of `all` / `any` / `not` / field comparisons (`==, !=, <, <=, >, >=, contains, in, is_empty, is_true, startswith, len()`,
   compare with a constant or another field). The validator type-checks the operator against the field type and the value against the operator (`E_PREDICATE_TYPE`,
   `E_PREDICATE_FIELD`, `E_PREDICATE_OP`). A route records, for every executed decision, each evaluated case with its predicate text and the operand values; the first matching case wins and
   later cases are not evaluated (the trace says so). The editor builds predicates with pickers; no Python is typed for routing, reducers, templates or memory policies.
5. **Bounded loops are three layers.** The route predicate (in-graph, e.g. `attempts < max_revisions` -> "return unresolved"), the **step limit** (`limits.maxSteps` -> LangGraph
   recursion limit; hitting it is recorded as `budget_exhausted(maxSteps)` and the run ends `completed` with `stoppedBy: recursion_limit`; the thread keeps its state), and run
   **budgets** (`maxModelCalls`, `maxTokens`, `maxSeconds`, `maxToolCalls`) checked before each node / model call (`stoppedBy: budget:<kind>`). A cycle without a conditional exit is a warning (`W_LOOP_STEP_LIMIT_ONLY`).
   Token budgets use provider-reported counts and fall back to the labelled estimate; unknown usage and unknown cost stay unknown in the record.
6. **Persistence and replay.** Checkpoints are LangGraph's `SqliteSaver` at `<workbench>/agent/checkpoints.sqlite`, keyed by thread id. An interrupt ends the worker process with run
   status `paused` (new state in the run ledger; `paused -> running` is atomic, so a double resume is rejected); `POST /api/runs/{id}/resume` starts a **new worker process** that
   loads the thread from the checkpoint and sends `Command(resume=...)`. A control-service crash therefore loses nothing; this is tested with a real SIGKILL and restart of a
   subprocess service. LangGraph re-executes the interrupted node from its start on resume (VISION 12.2): blocks put `interrupt()` before anything effectful, the trace marks
   such nodes `replayed`, and **protected effects** (external tool effects, long-term memory writes) go through an **effect ledger** (`effects` table in `memory.db`): the key is
   `thread:node:tool:hash(arguments)`; `claim -> perform -> complete`; a replay (resume, or a time-travel re-run from an earlier checkpoint) finds `done` and skips; a claim that
   never completed is reported `effect_uncertain` and not repeated. The step number is deliberately not part of the key because replay can run a node at a different step. Consequence
   (documented): the identical call in the same thread and node happens once; to repeat an effect on purpose the arguments must differ (e.g. a counter).
   External effects (`file_write`, `network`) always require an approval interrupt; turning it off is a validation error (`E_TOOL_APPROVAL`). File tools are bounded to a declared directory
   (resolved paths, symlinks and `..` cannot escape). The calculator parses arithmetic only (no names, attributes or calls).
7. **Model providers.** `ollama` (the tested live path; qwen3.5 models, `think` off by default), `anthropic` (`claude-sonnet-5-5` default; key by **secret reference** from
   `connectors/secrets`, never stored or returned; live test skipped without a key), and `fixture`, a scripted `BaseChatModel` used for control-flow tests. A fixture is labelled
   `FIXTURE` in validation (a warning), in every recorded call (`fixture: true`), in the API, the cards, the timeline and the context inspector; it reports no token usage. Only
   settings the provider supports are applied; ignored settings are recorded with the reason (e.g. Anthropic has no sampling seed). Usage is provider-reported or `unavailable`; cost is
   `0 (local inference)` for Ollama and `unknown` otherwise (no pricing table is claimed).
8. **Retrieval.** Corpus indexing is separate from per-question execution: an index (loader -> splitter -> embeddings -> FAISS flat inner product) is built under
   `<workbench>/agent/indexes/<id>/` only when files, splitter or embedding identity changed; chunk embeddings are cached by (embedding identity, text hash) so unchanged chunks are not
   re-embedded. Embedding providers: Ollama `nomic-embed-text` (pulled; `/api/embed` does not work with the qwen3.5 models) and `local_hash`, a deterministic hashed bag-of-words
   with a small stopword list, labelled **lexical, not semantic**; it is the default in tests. Scores are cosine similarities when normalized and say so. Every retrieval records included
   chunks with scores and the scored-but-cut chunks with the reason (threshold or k). Citations are checked against the retrieved chunks (`agent.citations`); a marker that is not a retrieved chunk is invalid.
9. **Memory is a visible subsystem, not a block.** Short-term memory is the thread's messages in the LangGraph checkpoint (native thread state, versioned, scoped by thread id). Long-term
   memory is `memory.db` (`records`: namespace, scope, kind, text, metadata, importance, timestamps, version, generated flag, source; soft delete). Every write records the producing node, run,
   evidence, old and new values, validation and scope (`memory_writes`); generated assertions are marked generated; an optional decision stage (interrupt) accepts, rejects or edits a proposal.
   A **memory policy** is an ordered pipeline of typed stage blocks (`retrieve`, `filter`, `rank`, `dedupe`, `budget`, `summarize`) edited visually. `run_policy` is a pure function of
   (policy, record universe, state, clock) and records, for every stored record, its status and a per-stage trail with a specific reason ("excluded by scope filter...", "ranked below the
   selected limit: rank 6 of 6, limit 3 (score 0.124 < cutoff 0.701)", "removed to meet the configured token budget: needs 38 tokens (estimate), 21 of 120 remain", "duplicate of ...",
   "replaced by summary ..."). The universe includes records outside the policy's scope, so a missing record can always be traced to the stage that dropped it (or to being outside the universe).
   Token counts for budgets are estimates (`ceil(chars/4)`), labelled as such. Deleting a stored record and removing it from an assembled context are separate: the delete response lists the recorded selections that still hold a snapshot.
10. **The context inspector records the request, not an inference of it.** For every model call the worker stores a `model_context` artifact (content-addressed, linked from the `model_call`
    event): the exact message list handed to the provider, the **segments** of each message (character ranges linked to prompt template + item, memory record id + selection id, retrieved chunk id + retrieval id
    + score, tool call id, conversation message), the resolved and ignored settings, provider-reported usage next to the labelled estimate, the model's context limit when the runtime reports it
    (else "unknown"), the reserved output allowance, the response, and the **excluded** items of every selection and retrieval that fed the call (found by the segments and, for empty selections, by following the graph
    from the messages field), each with stage and reason. Stored-but-unused records are never in the included set (tested as disjoint). The observability boundary is stated: provider-side formatting is not observable.
11. **Isolated preview (A32).** `preview_policy_edit` re-runs the edited policy on the *records recorded for that call* (the application keeps the whole universe and the clock), re-renders the prompt from the recorded
    prompt inputs, and diffs against the recorded request. It reads recorded data only: no chat model is called, no store/state/run/artifact/event is written (tested by comparing database and artifact snapshots, and by making the
    model function raise). A policy that uses Ollama embeddings would call the embeddings endpoint; the response flags it. A model-summary stage is skipped and said so. A faithful engine is tested: previewing the unedited policy
    reproduces the recorded context exactly, and previewing the edited one predicted the real rerun's context.
12. **Runs reuse the event store.** `POST /api/runs` accepts an agent graph (`AgentRunConfig`: thread id, input); events are `run_started` (libraries, limits, thread), `node_started` (step, replay), `state_update`
    (reads and per-field before/after with the reducer), `node_finished`, `route_taken`, `prompt_rendered`, `model_call`, `structured_attempt`, `retrieval`, `tool_call`, `memory_selection`, `memory_write`, `interrupt_raised`, `interrupt_resumed`,
    `run_paused`, `budget_exhausted`, `run_finished`, with run id, graph hash and a continuous sequence across resumes. Viewing a trace or a context makes no model call; **rerun** is a separate action that starts a new run on a new thread.
    Forking copies a checkpoint to a new thread with edits; the original thread is untouched.

## Consequences and known limits

- Model-driven tool selection (a model choosing among tools and arguments) is not implemented; tool blocks are explicit nodes with bounded capabilities. Subgraph nodes and cross-thread LangGraph `Store` are not implemented (long-term memory is `memory.db`).
- Token budgets in memory policies use the character estimate; no tokenizer is bundled. Cost needs a pricing table that is not configured.
- Comparing chunking strategies or embedding models side by side is possible by building indexes with different specs but there is no comparison view.
- Streaming of model tokens, async execution, and retention controls for checkpoints/records (expiry sweeps) are not implemented. Record `expires_at` is honoured by the retrieve stage.
- The Anthropic adapter is exercised by construction and parameter-mapping tests; no live call was made (no key).
