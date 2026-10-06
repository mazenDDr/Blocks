# ADR0056: offline physical erasure of one serving user

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded separately in HANDOFF§53.

Conversation reset (ADR0025) is logical: old checkpoints, traces and captured
inputs stay in the workbench. With named accounts (ADR0055) serving data has an
owner, so a person can ask for their data to be removed, and the system had no
way to do it.

Decision: `python -m maintenance.erase --workbench W --user U` (dry run by
default; `--apply --offline` to erase), built on the offline collector (ADR0050).

- Before any change, every database passes the collector's checks (integrity, no
  queued/running work, known CAS references present).
- Rows deleted from production.sqlite: the user's requests (so their recorded
  traces), labels and conversation actions, and counter/native conversation heads
  whose `[release, user, session]` scope names the user.
- The database is WAL-checkpointed (TRUNCATE), VACUUMed and checkpointed again,
  so deleted rows do not survive in free pages or the WAL.
- The blobs those rows named, and everything those blobs name, are erasure
  candidates. After the row deletion a fresh census of the whole workbench runs;
  candidates nothing else references are physically deleted, and candidates
  still referenced by another record are kept and listed with the reason, so
  erasure never breaks someone else's data. Other unreferenced garbage is left to
  the collector.

Not erased: copies outside the workbench (backups, sealed files, exports, MLflow/
W&B runs, other machines), audit log entries (paths, account names, times), source
or research runs someone recorded separately, and content another user's records
still reference. There is no online erasure and no erasure receipt stored in the
workbench (the report is printed).

Verification: 4 pytest cases on real native conversation checkpoints served
offline. Two users each hold a conversation containing a unique marker, plus
labels for one. A dry run reports the exact row counts and changes nothing, and
apply refuses without the offline attestation. After applying, a raw byte scan of
every workbench file (SQLite pages, WAL, CAS and everything else) finds the
erased user's marker nowhere, while the other user's data remains: their trace
list, conversation head and a new turn continuing from the correct revision.
A blob another record still names is retained and reported. CLI dry run and
user-name refusal are covered.
