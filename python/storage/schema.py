"""Transactional, ordered SQLite migrations with explicit ownership and downgrade guards.

Migration steps are trusted application SQL, never HTTP/package input. Each database
owns its transaction; this is not an atomic upgrade of a whole workbench.
"""
from __future__ import annotations

from contextlib import closing
from functools import lru_cache
from pathlib import Path
import random
import re
import sqlite3
import time

OWNERS = {name: 0x564F0000 + i for i, name in enumerate(
    ('meta', 'production', 'connections', 'studies', 'integrations', 'research',
     'memory', 'embeddings', 'cache-retention'), 1)}


class SchemaError(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(f'{code}: {message}')


def statements(script):
    """Split complete trusted statements without executescript's implicit COMMIT."""
    result, pending = [], ''
    for char in script:
        pending += char
        if char == ';' and sqlite3.complete_statement(pending):
            result.append(pending.strip())
            pending = ''
    if pending.strip():
        raise ValueError('Migration SQL must contain complete semicolon-terminated statements.')
    return tuple(result)


def _canonical(sql):
    # Preserve quoted literals/identifiers; normalize only unquoted SQL whitespace/case.
    parts = re.split(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\")", sql)
    return ''.join(p if i % 2 else re.sub(r'\s+', '', p).upper() for i, p in enumerate(parts)).rstrip(';')


def _objects(db):
    return {name: (kind, _canonical(sql)) for kind, name, sql in db.execute(
        "SELECT type,name,sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'")}


@lru_cache(maxsize=32)
def _reference(steps, version):
    with closing(sqlite3.connect(':memory:')) as reference:
        for step in steps[:version]:
            for sql in step:
                reference.execute(sql)
        return _objects(reference)


def _identity(db, kind, maximum):
    version = db.execute('PRAGMA user_version').fetchone()[0]
    owner = db.execute('PRAGMA application_id').fetchone()[0]
    if version < 0 or version > maximum:
        raise SchemaError('E_SCHEMA_VERSION', f'{kind} has version {version}; this code supports 0..{maximum}. No downgrade is performed.')
    if owner not in ((0, OWNERS[kind]) if version == 0 else (OWNERS[kind],)):
        raise SchemaError('E_SCHEMA_OWNER', f'Database ownership does not match {kind}.')
    return version


def _shape(db, expected, partial=False):
    actual = _objects(db)
    if any(name not in expected or definition != expected[name] for name, definition in actual.items()) or (not partial and set(actual) != set(expected)):
        raise SchemaError('E_SCHEMA_SHAPE', 'Database objects differ from the declared application schema. No data is rewritten.')


def guard(db, kind, steps):
    steps = tuple(tuple(step) for step in steps)
    version = _identity(db, kind, len(steps))
    # Legacy version0 may lack tables added by earlier CREATE IF NOT EXISTS bootstraps.
    _shape(db, _reference(steps, version or 1), partial=version == 0)
    return version


def migrate(db, kind, steps):
    """Adopt compatible version0 or apply sequential steps, atomically per database."""
    if kind not in OWNERS or not steps or any(not step for step in steps):
        raise ValueError('Declare a known owner and nonempty ordered migration steps.')
    steps = tuple(tuple(step) for step in steps)
    version = guard(db, kind, steps)
    if version == len(steps):
        return version
    if db.in_transaction:
        raise SchemaError('E_SCHEMA_TRANSACTION', 'Migration requires its own transaction before application writes.')
    db.execute('BEGIN IMMEDIATE')
    try:
        version = guard(db, kind, steps)  # another process may have upgraded while we waited
        for number in range(version, len(steps)):
            for sql in steps[number]:
                db.execute(sql)
            db.execute(f'PRAGMA user_version={number + 1}')
        db.execute(f'PRAGMA application_id={OWNERS[kind]}')
        _shape(db, _reference(steps, len(steps)))
        db.commit()
    except BaseException:
        db.rollback()
        raise
    return len(steps)


def enable_wal(db, timeout=30.0):
    """Switch to WAL, retrying SQLite's immediate SQLITE_BUSY.

    The switch holds SHARED and then needs EXCLUSIVE; if another connection's commit took PENDING in between, SQLite returns
    "database is locked" without calling the busy handler (deadlock avoidance). Re-running the statement is the remedy."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return db.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        except sqlite3.OperationalError as error:
            if "locked" not in str(error) or time.monotonic() >= deadline:
                raise
            time.sleep(0.02 + random.random() * 0.08)


def open_database(path, kind, schema, *, timeout=30, **kwargs):
    db = sqlite3.connect(path, timeout=timeout, **kwargs)
    try:
        migrate(db, kind, (statements(schema),))
        return db
    except BaseException:
        db.close()
        raise


EMBEDDINGS = 'CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec TEXT NOT NULL);'
INTEGRATIONS = 'CREATE TABLE IF NOT EXISTS records(kind TEXT, id TEXT, value TEXT, updated REAL, PRIMARY KEY(kind,id));'
CACHE_RETENTION = '''
CREATE TABLE IF NOT EXISTS policies (project TEXT PRIMARY KEY, config TEXT NOT NULL, revision INTEGER NOT NULL, next_due REAL NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS receipts (seq INTEGER PRIMARY KEY AUTOINCREMENT, project TEXT NOT NULL, revision INTEGER NOT NULL, started REAL NOT NULL, finished REAL NOT NULL, result TEXT, error TEXT);
'''


def definitions():
    # Import only for offline inventory. Runtime callers pass their existing schema.
    from artifact_store.store import _SCHEMA as meta
    from production.store import SCHEMA as production
    from connectors.registry import _SCHEMA as connections
    from studies.store import _SCHEMA as studies
    from research.records import SCHEMA as research
    from agent.memory import SCHEMA as memory
    return {'meta.db': ('meta', meta), 'production.sqlite': ('production', production),
            'connections.db': ('connections', connections), 'studies.db': ('studies', studies),
            'integrations.sqlite': ('integrations', INTEGRATIONS), 'research.sqlite': ('research', research),
            'agent/memory.db': ('memory', memory), 'agent/embed_cache.sqlite': ('embeddings', EMBEDDINGS),
            'cache-retention.sqlite': ('cache-retention', CACHE_RETENTION)}


def inspect(workbench):
    root = Path(workbench).expanduser().resolve()
    result = []
    for relative, (kind, schema) in definitions().items():
        path = root / relative
        row = {'database': relative, 'owner': kind, 'supportedVersion': 1, 'present': path.is_file()}
        if path.is_file():
            with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as db:
                row['version'] = guard(db, kind, (statements(schema),))
                row['applicationId'] = db.execute('PRAGMA application_id').fetchone()[0]
            row['migrationNeeded'] = row['version'] == 0
        result.append(row)
    return {'databases': result, 'excluded': ['agent/checkpoints.sqlite (native LangGraph)', 'tracker/provider databases'],
            'policy': 'Version1 adopts only exact known legacy objects and adds missing declared objects; native rows and CAS bytes are preserved. Each database migrates independently.'}
