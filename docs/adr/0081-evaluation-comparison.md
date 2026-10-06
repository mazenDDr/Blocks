# ADR0081: comparing two evaluations

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded in HANDOFF§89.

Evaluations (ADR0077/0078) produced pass rates, but deciding whether a change to a graph,
model or release helped needs a per-case comparison: two pass rates with overlapping
intervals hide which cases changed.

Decision: `GET /api/agent/evals/compare?a=&b=` and a "compare with" view in the Evaluate tab.

- Cases are matched on (case id, seed); only keys present in both are compared, and the
  counts of keys only in A or only in B are reported.
- Each shared key is `fixed` (fail → pass), `regressed` (pass → fail) or `same`.
- Significance: exact two-sided McNemar test (binomial on the discordant cases). With
  few discordant cases the test cannot show a difference, which the interpretation says.
- Works for research and release evaluations alike.

Also decided here: tables stored before ADR0079 are not recompressed. Their hashes are
referenced by run events and pinned serving manifests; rewriting them would change
recorded provenance. New runs store compressed tables; old ones stay as recorded.

Not provided: comparisons of more than two evaluations, effect sizes beyond counts,
comparison when the two evaluations used different checks for the same case id (the
comparison trusts ids; it reports the checks' outcomes, not whether checks match).

Verification: `tests/test_agent_eval.py` (counts, exact p-values 1.0 and 0.0078,
unmatched keys; API comparison of two real evaluations with a regression; 404 for an
unknown evaluation) and the `agent_eval` journey (second evaluation 6/6, comparison
4 → 6, 2 fixed, 0 regressed, p = 0.5).
