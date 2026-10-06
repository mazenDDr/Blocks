# ADR0045: additive native JSON agent serving

Status: accepted on Mac arm64 (2026-10-06); hosted provider verification unavailable.

The existing production agent adapters serve text and native conversations. Their
saved-version identities pin native agent/store sources, so changing those sources
would invalidate existing versions. Structured output needs its own object contract
and successful model-call evidence; a default object is insufficient.

Add `agent_json` / `__agent_json_graph__` as a separate adapter reusing the existing
native LangGraph execution and text-input/reference contracts. Do not change any
legacy pinned source. New JSON versions additionally pin the new adapter and shared
production integration sources, source graph/config/final state, successful native
structured events and model-context membership, flat native JSON schema, budgets,
installed Ollama model digest/runtime and dependencies. Registration and every use
verify actual completed END source evidence. Each prediction requires a successful
non-fixture native structured call and schema-valid finite object, runs from declared
turn defaults in a fresh native scratch checkpoint, and leaves research state intact.

Bound the contract: prompt/set_state and exactly one structured-output node; no
retrieval/indexes, tools/effects, long-term/thread memory or interrupts. Maximum16
nodes/32KiB graph,32state fields/8KiB defaults,1–4text inputs,12flat schema fields/
4KiB schema,8KiB output,25steps,30seconds,1–2model calls,4096total token budget and
128declared output tokens/call. Require local installed Ollama, think=false,
on_failure=fail, retries within the declared call budget. Do not download a model
or substitute a fixture provider. Releases are stateless/maxBatch1 only.

Capture-on retains actual state/control events/sent contexts with provenance;
capture-off retains hashes/usage without storing those full values in serving CAS.
Existing admission/deadline/cancellation, durable request idempotency, explicit
new-call replay and release routing apply. Object ground truth must independently
match the pinned native schema. Monitoring compares sorted-key finite canonical
JSON while preserving number representation; literal agreement is not semantic
correctness. Prediction drift measures canonical UTF8 byte lengths against the
recorded warmup, not a new model invocation. The one-source reference is weak
and labelled descriptive evidence.

Verification requires the native/full/live suites, unchanged legacy saved versions,
actual installed-Ollama browser source/register/warmup/deploy/request/context/replay/
labels/monitor, independent curl refusals, all original editor journeys and offline
backup with source deletion followed by restored native/JSON execution. The optional
`--json-agent` recovery journey requires the actual provider and fails without it.
Default hosted Linux CI has no Ollama; it retains native contracts/all original
browser/recovery checks and does not claim real JSON provider verification.
No dependency/database schema/weight migration or legacy retraining is required.
Wider providers/streaming, conversation JSON, retrieval/memory/tools/effects and
remote deployment remain unsupported. See HANDOFF§42 for actual evolving evidence.

Final Mac evidence:1235native tests pass,1skip/14live deselected;14actual Ollama
live tests pass;279module editor build/types/33Node tests/coverage pass. Actual
Chrome source/release/request/captured-replay/labels/monitor and16curl cases pass;
all12original editor journeys pass. Source-deleted71file/10DB/localtracker recovery
preserves old native state plus exact new JSON manifests/requests and executes a
fresh real model call. Existing pre-extension saved conversation version passes
its16native source identity and actual execution. Details/failures/readiness repair
are retained in HANDOFF§42; the whole VISION remains unfinished.
