# ADR0057: crash recovery of run workers

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded separately in HANDOFF§54.

Runs execute in spawned worker processes that write straight to the workbench, so a
run survives the control service going away. But if the worker itself dies (OOM
kill, crash, machine restart) or the control process crashes before the worker
starts, the run stays queued/preparing/running/cancelling forever. The UI shows a
phantom run, cancellation waits for a worker that does not exist, and backup and
garbage collection refuse the workbench because it has unfinished work.
Production requests already recover on restart (`recover_incomplete`); runs did not.

Decision: worker leases with heartbeats, and conservative reconciliation.

- Each worker process records a lease in meta.db (`run_leases`: run, pid, random
  nonce, start, heartbeat) and a daemon thread refreshes the heartbeat every 5 s
  for the life of the process. A later worker for the same run (an agent resume)
  replaces the lease; a heartbeat with another worker's nonce changes nothing.
- `ArtifactStore.reconcile_lost_workers()` fails an active run only when its
  newest sign of life (heartbeat, last event or last status change) is older than
  120 s. Status becomes `failed` with an `E_WORKER_LOST` message naming the
  silence and previous status, and a `run_finished` event with `recovered: true`
  and the worker pid is appended. Recorded events and artifacts are kept; nothing
  is resumed, retried or invented. Paused agent runs (waiting for a person, with no
  worker by design) are never touched. Reconciliation is idempotent.
- The control service reconciles at start-up and every 30 s while running, so a
  crash is cleared on restart and a worker killed during operation is cleared
  within about two and a half minutes.

Silence, not PID checks, is the signal: PIDs are reused and workers may run in
another process tree. The cost is that a worker blocked for more than 120 s
without releasing the GIL would be marked lost; its later writes would then be
refused by the status transitions. Not provided: automatic retry/resume of lost
runs, recovery of research studies beyond their existing resumable bookkeeping,
leases for the remote worker service's own client view, or a UI badge beyond the
failed status and error text.

Verification: 4 pytest cases. Store level: silent queued/preparing/running/
cancelling runs fail with evidence events while paused and terminal runs are
untouched, reconciliation is idempotent, and a fresh heartbeat or event keeps a run
alive with nonce-scoped heartbeats. Real processes: two CNN training workers start;
one is SIGKILLed mid-training; the live worker's heartbeat advances, and
reconciliation fails only the killed run (with its pid) and keeps its artifacts,
while the live run continues and is then cancelled normally. The control service
fails a crash-orphaned run at start-up.
