"""Trusted local directory snapshots. Stop *all* writers before calling create.

SQLite reserved locks and filesystem change detection supplement that requirement;
they cannot make a live control/worker/tracker tree an atomic online snapshot.
"""
from __future__ import annotations

from contextlib import ExitStack, closing
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path, PurePosixPath
import platform
import posixpath
import re
import shutil
import sqlite3
import stat
import tempfile
import time

FORMAT = "void-offline-workbench-v1"
LINK_FORMAT = "void-offline-workbench-v2"
HEX = re.compile(r"[0-9a-f]{64}\Z")
REQUIRED_DBS = {"meta.db", "production.sqlite", "connections.db", "studies.db", "integrations.sqlite",
                "agent/checkpoints.sqlite", "agent/memory.db", "agent/embed_cache.sqlite"}
POLICY = ("Offline trusted local backup. All writers must be stopped. Absolute paths and native identities are unchanged. "
          "External datasets, secrets, services, code and dependency environments are not bundled. "
          "Checksums detect corruption, not authenticity; no cross-version migration or online snapshot guarantee.")
LINK_POLICY = POLICY + " Internal relative links are inventoried without traversal; explicitly omitted external W&B debug logs are recorded."


class BackupError(Exception):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def fail(code, message):
    raise BackupError(code, message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(root, *, allow_links=False):
    """Never follow symlinks, including directory links. Preserve empty directories."""
    files, dirs, links = {}, [], {}
    for parent, names, children in os.walk(root, followlinks=False):
        for name in sorted(names + children):
            p = Path(parent) / name
            mode = p.lstat().st_mode
            rel = p.relative_to(root).as_posix()
            if stat.S_ISDIR(mode):
                dirs.append(rel)
            elif stat.S_ISREG(mode):
                files[rel] = p
            elif stat.S_ISLNK(mode) and allow_links:
                links[rel] = os.readlink(p)
            else:
                fail("E_BACKUP_FILE_TYPE", f"Only regular files/directories are supported: {rel}.")
    return dict(sorted(files.items())), sorted(dirs), dict(sorted(links.items()))


def tree(root):
    files, dirs, _ = inventory(root)
    return files, dirs


def external_wandb_log(rel, target):
    """Only the native W&B export's external diagnostic log may be omitted."""
    parts = PurePosixPath(rel).parts
    return (len(parts) == 7 and parts[:2] == ("trackers", "wandb") and HEX.fullmatch(parts[2])
            and parts[3] == "wandb" and parts[4].startswith("offline-run-")
            and parts[5:] == ("logs", "debug-core.log") and target.startswith("/"))


def link_records(raw, files, dirs, databases, *, omit_wandb_external_logs=False):
    records, omitted = {}, {}
    for rel, target in raw.items():
        safe_relative(rel)
        if not isinstance(target, str) or not target or "\\" in target or "\x00" in target:
            fail("E_BACKUP_LINK", f"Invalid symbolic link: {rel}.")
        if rel == "artifacts" or rel.startswith("artifacts/") or rel in REQUIRED_DBS or rel.endswith(("-wal", "-shm", "-journal")):
            fail("E_BACKUP_LINK", f"Links cannot replace owned databases or CAS: {rel}.")
        if target.startswith("/"):
            if omit_wandb_external_logs and external_wandb_log(rel, target):
                omitted[rel] = {"target": target, "reason": "external-wandb-debug-log"}
                continue
            fail("E_BACKUP_LINK", f"Absolute/external link refused: {rel}.")
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(rel), target))
        try:
            safe_relative(resolved)
        except BackupError:
            fail("E_BACKUP_LINK", f"Escaping symbolic link refused: {rel}.")
        # Normalization alone is insufficient: alias/../file would traverse
        # alias before the '..'. Check every intermediate component lexically.
        walked = list(PurePosixPath(rel).parent.parts)
        for i, part in enumerate(target.split("/")):
            if part in ("", "."):
                continue
            if part == "..":
                if not walked:
                    fail("E_BACKUP_LINK", f"Escaping symbolic link refused: {rel}.")
                walked.pop()
            else:
                walked.append(part)
            if i < len(target.split("/")) - 1 and walked and "/".join(walked) not in dirs:
                fail("E_BACKUP_LINK", f"Link traverses a non-directory or another link: {rel}.")
        # Membership is lexical in the no-follow inventory: chains, dangling
        # targets and paths traversing a directory link cannot be admitted.
        if (resolved not in files and resolved not in dirs) or resolved in databases:
            fail("E_BACKUP_LINK", f"Link target must be an inventoried ordinary file/directory, not a database: {rel}.")
        records[rel] = {"target": target, "resolved": resolved, "kind": "directory" if resolved in dirs else "file"}
    return records, omitted


def sqlite_file(path):
    with path.open("rb") as f:
        return f.read(16) == b"SQLite format 3\x00"


def connect(path, mode="ro"):
    db = sqlite3.connect(path.as_uri() + f"?mode={mode}", uri=True, timeout=.1)
    db.execute("PRAGMA trusted_schema=OFF")
    return db


def check_database(db, rel, artifacts):
    if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        fail("E_BACKUP_SQLITE", f"SQLite integrity check failed: {rel}.")
    # These are fixed owned schema/table identifiers; never SQL from a manifest.
    active = {"meta.db": [("runs", "status", "'queued','preparing','running','cancelling'")],
              "production.sqlite": [("requests", "state", "'queued','running'")],
              "studies.db": [("studies", "state", "'running','cancelling'"),
                             ("attempts", "status", "'running'")]}
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table, column, states in active.get(rel, []):
        if table in tables and db.execute(f"SELECT 1 FROM {table} WHERE {column} IN ({states}) LIMIT 1").fetchone():
            fail("E_BACKUP_ACTIVE", f"Unfinished work in {rel}/{table}; finish or recover it before backup.")
    references = {"meta.db": [("artifacts", "sha256"), ("node_cache", "sha256")],
                  "production.sqlite": [("records", "sha"), ("requests", "trace"),
                                        ("agent_sessions", "checkpoint"), ("conversation_actions", "result")]}
    for table, column in references.get(rel, []):
        if table in tables:
            for (sha,) in db.execute(f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL"):
                if not isinstance(sha, str) or not HEX.fullmatch(sha) or not (artifacts / sha).is_file():
                    fail("E_BACKUP_REFERENCE", f"Missing/invalid CAS reference in {rel}/{table}.")


def check_payload(root, database_names):
    files, dirs = tree(root)
    for rel, p in files.items():
        if rel.startswith("artifacts/"):
            sha = rel.removeprefix("artifacts/")
            if not HEX.fullmatch(sha) or digest(p) != sha:
                fail("E_BACKUP_CAS", f"Invalid CAS filename/content: {rel}.")
    for rel in database_names:
        try:
            with closing(connect(root / rel)) as db:
                check_database(db, rel, root / "artifacts")
        except sqlite3.Error as e:
            fail("E_BACKUP_SQLITE", f"Cannot validate {rel}: {type(e).__name__}.")
    return files, dirs


def new_destination(source, destination):
    dest = Path(destination).expanduser().absolute()
    if dest.is_symlink() or dest.exists():
        fail("E_BACKUP_DESTINATION", "Destination must not exist; existing data is never overwritten.")
    dest = dest.resolve()
    if dest.is_relative_to(source) or source.is_relative_to(dest):
        fail("E_BACKUP_DESTINATION", "Source and destination must be disjoint trees.")
    if not dest.parent.is_dir():
        fail("E_BACKUP_DESTINATION", "Destination parent directory must already exist.")
    return dest


def write_json(path, value):
    with path.open("w") as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    path.chmod(0o600)


def publish(stage, dest):
    # Reserve the name atomically, so two honest local operations cannot both
    # pass an exists() check and replace one another's destination directory.
    try:
        dest.mkdir(mode=0o700)
    except FileExistsError:
        fail("E_BACKUP_DESTINATION", "Destination appeared during publication; no overwrite.")
    try:
        stage.rename(dest)  # replace only our own empty reservation
    except BaseException:
        dest.rmdir()
        raise


def create(source, destination, *, offline=False, links="reject", omit_wandb_external_logs=False):
    if not offline:
        fail("E_BACKUP_OFFLINE", "Pass offline=True only after stopping all writers to this workbench.")
    root = Path(source).expanduser().resolve()
    if not root.is_dir() or not (root / "meta.db").is_file():
        fail("E_BACKUP_SOURCE", "Source must be a workbench containing meta.db.")
    dest = new_destination(root, destination)
    if links not in ("reject", "internal") or omit_wandb_external_logs and links != "internal":
        fail("E_BACKUP_LINK_POLICY", "Use links='internal' to preserve internal links and optionally omit external W&B debug logs.")
    files, dirs, raw_links = inventory(root, allow_links=links == "internal")
    # CAS is opaque immutable content, even when an artifact is itself a SQLite
    # file. Never normalize its pages/journal mode and thereby change its hash.
    databases = [rel for rel, p in files.items() if not rel.startswith("artifacts/") and sqlite_file(p)]
    for rel in REQUIRED_DBS & files.keys():
        if rel not in databases:
            fail("E_BACKUP_SQLITE", f"Required database is not SQLite: {rel}.")
    sidecars = {rel + suffix for rel in databases for suffix in ("-wal", "-shm", "-journal")}
    if any(rel.endswith(("-wal", "-shm", "-journal")) and rel not in sidecars for rel in files):
        fail("E_BACKUP_SQLITE", "Orphan/unrecognized SQLite sidecar; repair or remove it before backup.")
    records, omitted = link_records(raw_links, set(files) - sidecars, dirs, databases,
                                    omit_wandb_external_logs=omit_wandb_external_logs)
    stage = Path(tempfile.mkdtemp(prefix=".void-backup-", dir=dest.parent))
    try:
        payload = stage / "data"
        payload.mkdir(mode=0o700)
        for rel in dirs:
            (payload / rel).mkdir(mode=0o700)
        with ExitStack() as stack:
            # Hold every discovered DB's write reservation together. Backup through
            # a separate reader: backup on its own write transaction would block.
            for rel in databases:
                db = stack.enter_context(closing(connect(root / rel, "rw")))
                try:
                    db.execute("BEGIN IMMEDIATE")
                    check_database(db, rel, root / "artifacts")
                except sqlite3.Error as e:
                    fail("E_BACKUP_BUSY", f"Cannot reserve/validate {rel}: {type(e).__name__}.")
            baseline = {rel: digest(p) for rel, p in files.items() if rel not in sidecars and rel not in databases}
            for rel, p in files.items():
                target = payload / rel
                if rel in sidecars:
                    continue
                if rel in databases:
                    deadline = time.monotonic() + 120
                    def progress(status, remaining, total):
                        if time.monotonic() > deadline:
                            fail("E_BACKUP_BUSY", f"SQLite backup deadline exceeded: {rel}.")
                    with closing(sqlite3.connect(target)) as out:
                        with closing(connect(p)) as reader:
                            reader.backup(out, pages=128, progress=progress, sleep=.01)
                        # Self-contained snapshot; WAL committed pages are incorporated
                        # by backup(), so no transient WAL/SHM files are distributed.
                        out.execute("PRAGMA journal_mode=DELETE")
                else:
                    shutil.copyfile(p, target)
                target.chmod(0o600)
            after, after_dirs, after_links = inventory(root, allow_links=links == "internal")
            if raw_links != after_links or dirs != after_dirs or set(after) - sidecars != set(files) - sidecars or any(
                    digest(after[rel]) != sha or digest(payload / rel) != sha for rel, sha in baseline.items()):
                fail("E_BACKUP_CHANGED", "Source tree changed during backup; stop all writers and retry.")
            check_payload(payload, databases)
            copied, _ = tree(payload)
            policy = LINK_POLICY if links == "internal" else POLICY
            manifest = {"format": LINK_FORMAT if links == "internal" else FORMAT, "createdAt": time.time(), "sourceRoot": str(root), "policy": policy,
                        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                        "packages": {d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]}},
                        "directories": dirs, "databases": databases,
                        "files": {rel: {"sha256": digest(p), "size": p.stat().st_size} for rel, p in copied.items()}}
            if links == "internal":
                manifest.update(links=records, omittedLinks=omitted)
            write_json(stage / "manifest.json", manifest)
        publish(stage, dest)
        return {"backup": str(dest), "manifestSha256": digest(dest / "manifest.json"),
                "files": len(manifest["files"]), "databases": databases, "policy": policy,
                "links": len(records), "omittedLinks": omitted}
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        fail("E_BACKUP_MANIFEST", "Invalid relative path in manifest.")
    p = PurePosixPath(value)
    if p.is_absolute() or any(part in (".", "..") for part in value.split("/")) or str(p) != value:
        fail("E_BACKUP_MANIFEST", "Noncanonical/escaping path in manifest.")
    return value


def verify(backup, *, manifest_sha256=None):
    root = Path(backup).expanduser().resolve()
    if not root.is_dir():
        fail("E_BACKUP_SOURCE", "Backup directory does not exist.")
    all_files, all_dirs = tree(root)
    p = root / "manifest.json"
    if not p.is_file() or p.stat().st_size > 32 * 1024 * 1024:
        fail("E_BACKUP_MANIFEST", "Missing/oversized manifest.")
    if manifest_sha256 is not None and digest(p) != manifest_sha256:
        fail("E_BACKUP_MANIFEST", "Manifest differs from the separately recorded SHA256.")
    try:
        m = json.loads(p.read_text())
        if m["format"] not in (FORMAT, LINK_FORMAT) or not isinstance(m["files"], dict) or not isinstance(m["directories"], list) or not isinstance(m["databases"], list):
            raise ValueError()
        expected_policy = LINK_POLICY if m["format"] == LINK_FORMAT else POLICY
        if not isinstance(m["sourceRoot"], str) or not Path(m["sourceRoot"]).is_absolute() or m["policy"] != expected_policy:
            raise ValueError()
        if not all(isinstance(m["environment"][key], str) for key in ("python", "platform")) or not isinstance(m["environment"]["packages"], dict):
            raise ValueError()
        names = {safe_relative(rel) for rel in m["files"]}
        directories = {safe_relative(rel) for rel in m["directories"]}
        databases = {safe_relative(rel) for rel in m["databases"]}
        if not databases <= names or "meta.db" not in databases or len(databases) != len(m["databases"]) or len(directories) != len(m["directories"]):
            raise ValueError()
        if names & directories:
            raise ValueError()
        records, omitted = m.get("links", {}), m.get("omittedLinks", {})
        if not isinstance(records, dict) or not isinstance(omitted, dict) or m["format"] == FORMAT and (records or omitted):
            raise ValueError()
        link_names = {safe_relative(rel) for rel in records} | {safe_relative(rel) for rel in omitted}
        if set(records) & set(omitted) or link_names & (names | directories):
            raise ValueError()
        # Every parent must be an ordinary directory. This also rejects links
        # that would become ancestors of another payload/manifest path.
        for rel in names | directories | link_names:
            parent = posixpath.dirname(rel)
            if parent and parent not in directories:
                raise ValueError()
        checked, checked_omitted = link_records({rel: value["target"] for rel, value in (records | omitted).items()},
                                               names, directories, databases, omit_wandb_external_logs=True)
        if checked != records or checked_omitted != omitted:
            raise ValueError()
        payload = root / "data"
        if set(all_files) != {"manifest.json"} | {"data/" + rel for rel in names} or set(all_dirs) != {"data"} | {"data/" + rel for rel in directories}:
            fail("E_BACKUP_INTEGRITY", "Backup inventory differs from manifest.")
        actual_databases = {rel for rel in names if not rel.startswith("artifacts/") and sqlite_file(payload / rel)}
        if actual_databases != databases or not (REQUIRED_DBS & names) <= databases:
            fail("E_BACKUP_MANIFEST", "SQLite inventory differs from manifest.")
        for rel, item in m["files"].items():
            f = payload / rel
            if not HEX.fullmatch(item["sha256"]) or type(item["size"]) is not int or item["size"] < 0:
                raise ValueError()
            if f.stat().st_size != item["size"] or digest(f) != item["sha256"]:
                fail("E_BACKUP_INTEGRITY", f"File checksum/size mismatch: {rel}.")
        check_payload(payload, sorted(databases))
    except (KeyError, TypeError, ValueError) as e:
        fail("E_BACKUP_MANIFEST", f"Malformed/unsupported manifest: {type(e).__name__}.")
    return m


def restore(backup, destination, *, trusted=False, manifest_sha256=None):
    if not trusted:
        fail("E_BACKUP_TRUST", "Restore only a trusted local backup; native artifacts may execute code when later loaded.")
    root = Path(backup).expanduser().resolve()
    dest = new_destination(root, destination)
    original_manifest = digest(root / "manifest.json")
    m = verify(root, manifest_sha256=manifest_sha256)
    stage = Path(tempfile.mkdtemp(prefix=".void-restore-", dir=dest.parent))
    try:
        for rel in sorted(m["directories"]):
            (stage / rel).mkdir(mode=0o700)
        for rel, info in m["files"].items():
            target = stage / rel
            shutil.copyfile(root / "data" / rel, target)
            target.chmod(0o600)
            if target.stat().st_size != info["size"] or digest(target) != info["sha256"]:
                fail("E_BACKUP_CHANGED", "Backup changed during restore.")
        check_payload(stage, m["databases"])
        if digest(root / "manifest.json") != original_manifest:
            fail("E_BACKUP_CHANGED", "Manifest changed during restore.")
        for rel, info in m.get("links", {}).items():
            (stage / rel).symlink_to(info["target"], target_is_directory=info["kind"] == "directory")
        # Links are created only after all ordinary bytes/DBs are validated.
        _, _, restored_links = inventory(stage, allow_links=True)
        if restored_links != {rel: info["target"] for rel, info in m.get("links", {}).items()}:
            fail("E_BACKUP_CHANGED", "Restored link inventory differs from manifest.")
        publish(stage, dest)
        return {"workbench": str(dest), "sourceRoot": m["sourceRoot"], "policy": m["policy"],
                "manifestSha256": original_manifest, "files": len(m["files"]),
                "links": len(m.get("links", {})), "omittedLinks": m.get("omittedLinks", {}),
                "samePython": m["environment"]["python"] == platform.python_version(),
                "samePlatform": m["environment"]["platform"] == platform.platform()}
    finally:
        if stage.exists():
            shutil.rmtree(stage)
