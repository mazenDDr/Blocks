# ADR0078: evaluation repeats by seed, cases from files, and evaluating deployed releases

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded in HANDOFF§85.

ADR0077 evaluated a research graph once per case, with cases typed as JSON, and could
not evaluate what is actually deployed.

Decision:

- **Seeds.** `seeds` (up to 5, unique; cases × seeds ≤ 400) on an agent evaluation.
  Each seed is written into every model block's `seed` (chat and structured output);
  one child run per case and seed (`<eval>-sNcNNN`, config records the seed). The
  report adds per-seed pass rates with their own Wilson intervals over cases and which
  cases always pass, never pass or vary with the seed. Runs of one case are not
  independent, so the overall count is reported but per-seed rates are the measure.
- **Files.** The editor loads cases from a `.json` array or a `.jsonl` file (one case
  per line, line numbers in errors); the server receives the same JSON.
- **Deployed releases.** `POST /api/production/releases/{id}/evaluate` (admin when
  accounts are on; the release must be routed). Each case is one real serving request
  through the route, recorded with its trace, as a user and session unique to the
  evaluation and case (`ev<id>-NNN`), so conversation state and release memory are
  never shared between cases or with real users. Checks read `output` (the served
  prediction) or `output.<key>`; any family works, including tabular/model releases.
  Runs in a control-process thread as a run of kind `release_eval` with the same
  events and report; a control restart mid-evaluation is recovered as lost
  (ADR0057). Editor: Production → Release → "Evaluate this deployed release".

Not provided: seeds for releases (a release pins its graph), parallel cases, comparing
two evaluations, deleting evaluation users' traces automatically (they are ordinary
recorded requests; per-user erasure applies).

Verification: `tests/test_agent_eval.py` (seeded graph copy, per-seed report, duplicate
seeds refused; a deployed memory release evaluated over the API with one designed
failure, every case recalling 0 notes as its own user, a real user's memory unchanged,
refusal when not deployed), journeys `agent_eval` (cases from a real .jsonl file, two
seeds, 4/6, per-seed table) and `release_memory` (release evaluation 2/3 from the
Release tab, real users' memory unchanged).
