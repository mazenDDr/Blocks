"""Offline physical erasure of one serving user's production data (ADR 0056).

Deletes the user's rows from production.sqlite: requests (and so their recorded traces), labels, conversation actions, counter
sessions and native conversation heads (scopes are [release, user, session]). The database is then checkpointed and VACUUMed
so deleted rows do not survive in free pages or the WAL. Finally, CAS blobs that those rows referenced (directly or through
other blobs) are physically deleted if, after the row deletion, nothing else in the workbench still references them; the
offline collector's over-approximating census decides that (ADR 0050). Blobs another record still needs are kept and reported.

Not erased: copies outside this workbench (backups, sealed files, exports, tracker runs, logs on other machines), the audit log's
paths, research/source runs a person recorded separately, and anything a different user's data references.
"""
from __future__ import annotations

from contextlib import closing
import json
import re
import sqlite3
from pathlib import Path

from storage.schema import guard, statements
from production.store import SCHEMA as PRODUCTION_SCHEMA

from .cas_gc import GcError, blob_references, mark

USER = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
TABLES = (("requests", "user"), ("labels", "user"), ("conversation_actions", "user"))
SCOPED = ("sessions", "agent_sessions")


def _scope_user(scope):
    try:
        value = json.loads(scope)
        return value[1] if isinstance(value, list) and len(value) == 3 else None
    except (TypeError, ValueError):
        return None


def erase_user(workbench, user: str, *, apply=False, offline=False) -> dict:
    if not USER.fullmatch(user or ""):
        raise GcError("E_ERASE_USER", "User must be 1-64 of A-Za-z0-9_-.")
    if apply and not offline:
        raise GcError("E_GC_OFFLINE", "Pass offline=True only after stopping all writers to this workbench.")
    root = Path(workbench).resolve()
    db_path = root / "production.sqlite"
    if not db_path.is_file():
        raise GcError("E_ERASE_STORE", "This workbench has no production database.")
    mark(root)  # integrity, active-work and reference checks of every database before any change
    with closing(sqlite3.connect(db_path, timeout=1)) as db:
        guard(db, "production", (statements(PRODUCTION_SCHEMA),))
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        rows, referenced = {}, set()
        for table, column in TABLES:
            if table in tables:
                found = db.execute(f"SELECT * FROM {table} WHERE {column}=?", (user,)).fetchall()
                rows[table] = len(found)
                referenced |= blob_references(root, json.dumps(found, default=str).encode())
        scopes = {}
        for table in SCOPED:
            if table in tables:
                found = [r for r in db.execute(f"SELECT * FROM {table}").fetchall() if _scope_user(r[0]) == user]
                scopes[table] = [r[0] for r in found]
                rows[table] = len(found)
                referenced |= blob_references(root, json.dumps(found, default=str).encode())
        report = {"user": user, "applied": False, "rows": rows, "referencedBlobs": len(referenced)}
        if not apply:
            return report
        with db:
            for table, column in TABLES:
                if table in tables:
                    db.execute(f"DELETE FROM {table} WHERE {column}=?", (user,))
            for table, keys in scopes.items():
                db.executemany(f"DELETE FROM {table} WHERE scope=?", [(k,) for k in keys])
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        db.execute("VACUUM")
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    census = mark(root)
    erased = sorted(referenced - census["kept"])
    retained = sorted(referenced & census["kept"])
    for sha in erased:
        (census["artifacts"] / sha).unlink()
    return {**report, "applied": True, "erasedBlobs": erased, "retainedBlobs": retained,
            "retainedReason": "still referenced by other workbench records after the user's rows were deleted" if retained else None,
            "note": "Copies outside this workbench (backups, sealed files, exports, trackers) are not erased."}


def main(argv=None):
    import argparse
    import sys
    parser = argparse.ArgumentParser(prog="python -m maintenance.erase", description="Erase one serving user's production data (offline).")
    parser.add_argument("--workbench", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--apply", action="store_true", help="delete rows, VACUUM and remove now-unreferenced blobs")
    parser.add_argument("--offline", action="store_true", help="attest that all control/worker/tracker/repository writers are stopped")
    args = parser.parse_args(argv)
    try:
        report = erase_user(args.workbench, args.user, apply=args.apply, offline=args.offline)
    except GcError as error:
        print(json.dumps({"error": error.code, "message": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["erase_user"]
