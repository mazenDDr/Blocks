"""Conservative offline garbage collection of the workbench content-addressed store.

Marking over-approximates references: every 64-hex window found in any workbench file
outside ``artifacts/`` keeps the blob of that name, and kept blobs are scanned the same way.
SQLite databases are read cell by cell through SQLite (a raw scan could miss a reference
split across overflow pages, and pending WAL frames are only visible through SQLite).
Other files, file names and link targets are scanned as raw bytes. A blob is a candidate
only if nothing marks it and it is older than the grace period. Deletion requires the
same offline attestation as backup: every writer to this workbench must be stopped.

References inside compressed or encrypted bytes cannot be seen; workbench writers store
CAS references as plain hex text, which the integrated recovery check exercises.
"""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

from workbench_backup.core import BackupError, check_database, connect, sqlite_file

HEX_RUN = re.compile(rb"[0-9a-f]{64,}")
NAME = re.compile(r"[0-9a-f]{64}\Z")
CHUNK = 1024 * 1024
LIST_LIMIT = 1000
POLICY = ("Offline conservative CAS collection. All writers must be stopped. Any 64-hex text in workbench files, "
          "SQLite cells, names, link targets or kept blobs keeps that blob. Compressed/encrypted references are not "
          "visible. Deletion is physical and not reversible except from a backup taken before collection.")


class GcError(Exception):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def _windows(data: bytes, names: set[str], found: set[str]) -> None:
    for run in HEX_RUN.findall(data):
        if len(run) == 64:
            text = run.decode()
            if text in names:
                found.add(text)
        else:
            # A longer hex run may embed a reference anywhere; keep every matching window.
            text = run.decode()
            for i in range(len(text) - 63):
                if text[i:i + 64] in names:
                    found.add(text[i:i + 64])


def _scan_file(path: Path, names: set[str], found: set[str]) -> None:
    tail = b""
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            data = tail + chunk
            _windows(data, names, found)
            # A window crossing into the next chunk needs at most 63 preceding hex characters.
            m = re.search(rb"[0-9a-f]{1,63}\Z", data)
            tail = m.group(0) if m else b""


def _scan_sqlite(path: Path, names: set[str], found: set[str]) -> None:
    with closing(connect(path)) as db:
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for (sql,) in db.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL"):
            _windows(sql.encode(), names, found)
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            for row in db.execute(f"SELECT * FROM {quoted}"):
                for value in row:
                    if isinstance(value, str):
                        _windows(value.encode("utf-8", "surrogatepass"), names, found)
                    elif isinstance(value, bytes):
                        _windows(value, names, found)


def _check_databases(root: Path, artifacts: Path, databases: list[str]) -> None:
    for rel in databases:
        with closing(connect(root / rel)) as db:
            try:
                check_database(db, rel, artifacts)
            except BackupError as error:
                raise GcError(error.code.replace("E_BACKUP", "E_GC"), str(error)) from error


def mark(workbench) -> dict:
    """Census of the store: every blob, every blob a reference keeps (transitively), partial writes and scanned inputs."""
    root = Path(workbench).resolve()
    artifacts = root / "artifacts"
    if not artifacts.is_dir() or artifacts.is_symlink():
        raise GcError("E_GC_STORE", "No owned artifacts directory in this workbench.")
    blobs, partial = {}, {}
    for entry in os.scandir(artifacts):
        st = entry.stat(follow_symlinks=False)
        if not stat.S_ISREG(st.st_mode):
            raise GcError("E_GC_FILE_TYPE", f"Only regular CAS files are supported: artifacts/{entry.name}.")
        if NAME.fullmatch(entry.name):
            blobs[entry.name] = st
        elif entry.name.endswith(".tmp"):
            partial[entry.name] = st
        else:
            raise GcError("E_GC_FILE_TYPE", f"Unexpected CAS file name: artifacts/{entry.name}.")
    names = set(blobs)
    found: set[str] = set()
    databases, scanned = [], 0
    for parent, dirs, files in os.walk(root, followlinks=False):
        rel_parent = Path(parent).relative_to(root)
        if rel_parent == Path("."):
            dirs[:] = [d for d in dirs if d != "artifacts"]
        for name in dirs + files:
            path = Path(parent) / name
            _windows(str(path.relative_to(root)).encode(), names, found)
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                _windows(os.readlink(path).encode(), names, found)
            elif stat.S_ISREG(mode) and name in files:
                rel = path.relative_to(root).as_posix()
                if rel.endswith(("-wal", "-shm", "-journal")):
                    continue  # read through SQLite with the owning database
                if sqlite_file(path):
                    databases.append(rel)
                    _scan_sqlite(path, names, found)
                else:
                    _scan_file(path, names, found)
                scanned += 1
    _check_databases(root, artifacts, databases)
    # Transitive marking: kept blobs may themselves name other blobs.
    pending, kept = list(found), set(found)
    while pending:
        sha = pending.pop()
        more: set[str] = set()
        _scan_file(artifacts / sha, names, more)
        for item in more - kept:
            kept.add(item)
            pending.append(item)
    return {"root": root, "artifacts": artifacts, "blobs": blobs, "partial": partial, "kept": kept,
            "databases": sorted(databases), "scanned": scanned}


def blob_references(workbench, text: bytes) -> set[str]:
    """Existing blobs named in `text`, plus everything they name transitively."""
    artifacts = Path(workbench).resolve() / "artifacts"
    names = {e.name for e in os.scandir(artifacts) if NAME.fullmatch(e.name)}
    found: set[str] = set()
    _windows(text, names, found)
    pending = list(found)
    while pending:
        more: set[str] = set()
        _scan_file(artifacts / pending.pop(), names, more)
        for item in more - found:
            found.add(item)
            pending.append(item)
    return found


def collect(workbench, *, apply=False, offline=False, grace_seconds=86400.0, now=None) -> dict:
    """Mark from every workbench file, then report (or, with apply+offline, delete) unreferenced old blobs."""
    if apply and not offline:
        raise GcError("E_GC_OFFLINE", "Pass offline=True only after stopping all writers to this workbench.")
    if not 0 <= grace_seconds <= 365 * 86400:
        raise GcError("E_GC_GRACE", "Grace period must be 0–365 days.")
    m = mark(workbench)
    artifacts, blobs, partial, kept, databases, scanned = m["artifacts"], m["blobs"], m["partial"], m["kept"], m["databases"], m["scanned"]
    now = time.time() if now is None else now
    names = set(blobs)
    unreferenced = sorted(names - kept)
    candidates = [s for s in unreferenced if now - blobs[s].st_mtime >= grace_seconds]
    stale_partial = sorted(n for n, st in partial.items() if now - st.st_mtime >= grace_seconds)
    listing = candidates + stale_partial
    report = {"policy": POLICY, "applied": False, "graceSeconds": grace_seconds, "scannedFiles": scanned,
              "databases": sorted(databases), "blobs": len(blobs), "referenced": len(kept),
              "unreferenced": len(unreferenced), "withinGrace": len(unreferenced) - len(candidates),
              "candidates": len(candidates), "candidateBytes": sum(blobs[s].st_size for s in candidates),
              "stalePartialFiles": len(stale_partial), "stalePartialBytes": sum(partial[n].st_size for n in stale_partial),
              "listSha256": hashlib.sha256(json.dumps(listing).encode()).hexdigest(),
              "list": listing[:LIST_LIMIT], "listTruncated": len(listing) > LIST_LIMIT}
    if apply:
        for name in listing:
            (artifacts / name).unlink()
        report["applied"] = True
    return report


def main(argv=None):
    import argparse
    import sys
    parser = argparse.ArgumentParser(prog="python -m maintenance.cas_gc", description=__doc__.splitlines()[0])
    parser.add_argument("--workbench", required=True)
    parser.add_argument("--grace-hours", type=float, default=24.0, help="keep unreferenced blobs younger than this (default 24)")
    parser.add_argument("--apply", action="store_true", help="physically delete the reported candidates")
    parser.add_argument("--offline", action="store_true", help="attest that all control/worker/tracker/repository writers are stopped")
    args = parser.parse_args(argv)
    try:
        report = collect(args.workbench, apply=args.apply, offline=args.offline, grace_seconds=args.grace_hours * 3600)
    except GcError as error:
        print(json.dumps({"error": error.code, "message": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
