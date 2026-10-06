# ADR0077: agent evaluation runs on labelled cases

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded in HANDOFF§84.

Agent graphs could be run and inspected one input at a time, but there was no way to
measure how often a graph produces the right answer: every serving and provider ADR
listed "no quality benchmark".

Decision: an evaluation run kind, `agent_eval` (`worker/agent_eval.py`).

- Submitted through `POST /api/runs` with an agent graph and
  `config: {kind: "agent_eval", name, cases: [...]}` (1–200 cases, unique ids). Each
  case has an `input` and 1–8 checks on fields of the final state (dotted paths into
  objects): `equals`, `contains`, `not_contains`, `regex`, `one_of`, `number_close`
  (first number in a text, absolute tolerance); case-insensitive unless declared.
- The worker runs each case as a real child agent run (`<eval>-cNNN`, fresh thread,
  config marked `evaluationOf`), recorded and inspectable like any run. A case passes
  only if its run completed and every check passed. Nothing is graded by a model.
- The evaluation records one `eval_case` event per case (checks with expected and
  observed values, latency, model calls, output tokens) and an `agent_eval_report`
  artifact: passed / cases, pass rate, Wilson 95% interval over cases, failed checks,
  status counts. Cancellation stops between cases.
- `GET /api/agent/evals/{id}`; editor: Agent → Evaluate (cases JSON, prefilled from an
  example's `evaluationCases`; score with interval; failed checks expected → observed;
  open each case's run).

Measured (local Ollama, temperature 0, seed 7, think off; `benchmarks/agent_eval_quality.py`,
example `evaluation_arithmetic`, 20 SYNTHETIC cases with computable answers):
qwen3.5:0.8b 12/20 (60%, Wilson 95% 39–78%): arithmetic 8/12, capitalisation 4/4,
reversal 0/4; qwen3.5:2b 16/20 (80%, 58–92%): arithmetic 12/12, capitalisation 4/4,
reversal 0/4. One run per model; pass rates over these cases only.

Not provided: model-graded or semantic checks, multiple seeds/repeats per case,
parallel case execution, comparison views between evaluations, evaluation of served
releases (research graphs only), datasets from files.

Verification: `tests/test_agent_eval.py` (10 check cases, Wilson values, an API
evaluation over real child runs with one designed failure, refusals; live: two cases on
qwen3.5:2b), browser journey `tools/editor_agent_eval_smoke.py`, and the live benchmark.
