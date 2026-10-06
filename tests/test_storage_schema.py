"""Actual legacy SQLite adoption, rollback, ownership and process-open races."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest
from artifact_store import ArtifactStore
from storage.schema import (steps_of, SchemaError, OWNERS, definitions, guard, inspect, migrate,
                            open_database, statements)


def rows(db):
    names=[r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    return {name:db.execute(f'SELECT * FROM "{name}" ORDER BY rowid').fetchall() for name in names}


def actual_open(root, kind):
    if kind=='meta':return ArtifactStore(root)
    if kind=='production':
        from production.store import ProductionStore
        return ProductionStore(ArtifactStore(root))
    if kind=='connections':
        from connectors.registry import ConnectionRegistry
        return ConnectionRegistry(root)
    if kind=='studies':
        from studies.store import StudyStore
        return StudyStore(root)
    if kind=='integrations':
        from scale.state import State
        return State(root)
    if kind=='research':
        from research.records import Records
        return Records(ArtifactStore(root))
    if kind=='memory':
        from agent.memory import MemoryStore
        return MemoryStore(root)
    if kind=='embeddings':
        from agent.index import EmbedCache
        return EmbedCache(root/'agent/embed_cache.sqlite')
    if kind=='cache-retention':
        from maintenance.cache_retention import CacheRetention
        owner=CacheRetention(ArtifactStore(root))
        with owner.database():pass
        return owner
    raise AssertionError(kind)


LEGACY_RECORDS={
 'meta':("INSERT INTO runs VALUES(?,?,?,?,?,?,?)",('SYNTHETIC-existing','a'*64,'completed','{}',None,1.,2.)),
 'production':("INSERT INTO records VALUES(?,?,?,?)",('version','a'*64,'a'*64,1.)),
 'connections':("INSERT INTO connections VALUES(?,?,?,?,?,?,?,?)",('synthetic','SYNTHETIC','postgres','{}','{}',1.,2.,None)),
 'studies':("INSERT INTO studies VALUES(?,?,?,?,?,?,?)",('synthetic',1.,2.,'{}','completed',None,0)),
 'integrations':("INSERT INTO records VALUES(?,?,?,?)",('synthetic','existing','{}',1.)),
 'research':("INSERT INTO annotations VALUES(?,?,?)",('synthetic','a'*64,1)),
 'memory':("INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",('synthetic','local','thread','semantic','SYNTHETIC existing native record','{}',1.,1.,2.,None,0,1,'{}',None)),
 'embeddings':("INSERT INTO emb VALUES(?,?)",('synthetic','[0.25,0.75]')),
 'cache-retention':("INSERT INTO policies VALUES(?,?,?,?,?)",('synthetic','{}',1,999.,1.)),
}


@pytest.mark.parametrize('relative,definition',list(definitions().items()))
def test_actual_store_adopts_legacy_without_rewriting_rows_or_native_objects(tmp_path,relative,definition):
    kind,schema=definition;path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True)
    legacy=schema if isinstance(schema,str) else schema[0]  # version0 legacy files have the original (step 1) shape
    with closing(sqlite3.connect(path)) as db:
        db.executescript(legacy);db.execute(*LEGACY_RECORDS[kind]);db.commit();before=rows(db)
        objects=db.execute("SELECT type,name,sql FROM sqlite_master ORDER BY name").fetchall()
        assert db.execute('PRAGMA user_version').fetchone()[0]==0
    actual_open(tmp_path,kind)
    with closing(sqlite3.connect(path)) as db:
        added={r[1] for r in db.execute("SELECT type,name,sql FROM sqlite_master")}-{o[1] for o in objects}
        assert {k:v for k,v in rows(db).items() if k not in added}==before and all(not rows(db).get(k) for k in added)
        assert [o for o in db.execute("SELECT type,name,sql FROM sqlite_master ORDER BY name").fetchall() if o[1] not in added]==objects
        assert db.execute('PRAGMA user_version').fetchone()[0]==len(steps_of(schema))
        assert db.execute('PRAGMA application_id').fetchone()[0]==OWNERS[kind]
    actual_open(tmp_path,kind)
    with closing(sqlite3.connect(path)) as db:assert {k:v for k,v in rows(db).items() if k not in added}==before


@pytest.mark.parametrize('relative,definition',list(definitions().items()))
def test_every_actual_store_refuses_future_versions_before_sql_or_file_mutation(tmp_path,relative,definition):
    kind,schema=definition;path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True)
    with closing(sqlite3.connect(path)) as db:
        for step in steps_of(schema):
            for sql in step:db.execute(sql)
        db.execute(f'PRAGMA user_version={len(steps_of(schema))+1}');db.commit()
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(SchemaError) as caught:actual_open(tmp_path,kind)
    assert caught.value.code=='E_SCHEMA_VERSION'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before


@pytest.mark.parametrize('foreign',[True,False])
def test_unknown_or_incompatible_legacy_objects_refuse_without_stamping(tmp_path,foreign):
    path=tmp_path/'meta.db'
    with closing(sqlite3.connect(path)) as db:
        db.execute('CREATE TABLE '+('foreign_data' if foreign else 'runs')+'(value TEXT)');db.commit()
    before=path.read_bytes()
    with pytest.raises(SchemaError) as caught:ArtifactStore(tmp_path)
    assert caught.value.code=='E_SCHEMA_SHAPE' and path.read_bytes()==before
    with closing(sqlite3.connect(path)) as db:assert db.execute('PRAGMA user_version').fetchone()[0]==0


def test_partial_known_legacy_adds_missing_declared_tables_and_preserves_rows(tmp_path):
    schema=definitions()['meta.db'][1];path=tmp_path/'meta.db'
    with closing(sqlite3.connect(path)) as db:
        db.execute(statements(schema)[0]);db.execute(*LEGACY_RECORDS['meta']);db.commit()
    store=ArtifactStore(tmp_path)
    assert store.get_run('SYNTHETIC-existing')['status']=='completed'
    assert store.node_cache_entries()==[]
    with closing(sqlite3.connect(path)) as db:assert guard(db,'meta',(statements(schema),))==1


def test_transactional_second_version_and_constraint_failure_rollback(tmp_path):
    baseline=(('CREATE TABLE emb(key TEXT PRIMARY KEY, vec TEXT NOT NULL);',),)
    steps=(*baseline,('CREATE TABLE upgrade_receipt(id INTEGER PRIMARY KEY);','CREATE UNIQUE INDEX vec_unique ON emb(vec);'))
    with closing(sqlite3.connect(tmp_path/'data.db')) as db:
        migrate(db,'embeddings',baseline)
        db.executemany('INSERT INTO emb VALUES(?,?)',[('a','same'),('b','same')]);db.commit()
        before=rows(db)
        with pytest.raises(sqlite3.IntegrityError):migrate(db,'embeddings',steps)
        assert rows(db)==before and guard(db,'embeddings',baseline)==1
        assert db.execute("SELECT name FROM sqlite_master WHERE name='upgrade_receipt'").fetchall()==[]
        db.execute("DELETE FROM emb WHERE key='b'");db.commit()
        assert migrate(db,'embeddings',steps)==2
        assert rows(db)=={'emb':[('a','same')],'upgrade_receipt':[]}
        with pytest.raises(SchemaError,match='No downgrade'):guard(db,'embeddings',baseline)


def test_wrong_owner_and_declared_current_shape_refuse(tmp_path):
    schema=definitions()['meta.db'][1];path=tmp_path/'meta.db'
    with closing(open_database(path,'meta',schema)) as db:
        with pytest.raises(SchemaError) as caught:guard(db,'production',(statements(schema),))
        assert caught.value.code=='E_SCHEMA_OWNER'
        db.execute('DROP TABLE run_leases');db.commit()
    with pytest.raises(SchemaError,match='E_SCHEMA_SHAPE'):ArtifactStore(tmp_path)


def test_concurrent_real_processes_adopt_once_and_preserve_both_writes(tmp_path):
    schema=definitions()['meta.db'][1]
    with closing(sqlite3.connect(tmp_path/'meta.db')) as db:
        db.executescript(schema);db.execute(*LEGACY_RECORDS['meta']);db.commit()
    script='from artifact_store import ArtifactStore;import sys;s=ArtifactStore(sys.argv[1]);s.create_run(sys.argv[2],"a"*64,{})'
    workers=[subprocess.Popen([sys.executable,'-c',script,str(tmp_path),f'SYNTHETIC-worker-{i}'],stdout=subprocess.PIPE,stderr=subprocess.PIPE) for i in range(2)]
    for worker in workers:
        stdout,stderr=worker.communicate(timeout=30);assert worker.returncode==0,(stdout,stderr)
    store=ArtifactStore(tmp_path);assert len(store.list_runs())==3
    with closing(sqlite3.connect(tmp_path/'meta.db')) as db:assert guard(db,'meta',(statements(schema),))==1


def test_readonly_inventory_offline_cli_preserves_native_checkpoint_and_cas(tmp_path):
    path=tmp_path/'meta.db';schema=definitions()['meta.db'][1]
    with closing(sqlite3.connect(path)) as db:db.executescript(schema)
    checkpoint=tmp_path/'agent/checkpoints.sqlite';checkpoint.parent.mkdir()
    with closing(sqlite3.connect(checkpoint)) as db:
        db.execute('CREATE TABLE native_provider(value TEXT)');db.execute('PRAGMA user_version=73');db.commit()
    native=checkpoint.read_bytes();before=path.read_bytes();report=inspect(tmp_path)
    assert report['databases'][0]['version']==0 and path.read_bytes()==before
    absent=subprocess.run([sys.executable,'-m','storage','migrate','--workbench',str(tmp_path)],capture_output=True,text=True)
    assert absent.returncode==2 and path.read_bytes()==before
    actual=subprocess.run([sys.executable,'-m','storage','migrate','--workbench',str(tmp_path),'--offline'],capture_output=True,text=True)
    assert actual.returncode==0,actual.stderr
    assert json.loads(actual.stdout)['after']['databases'][0]['version']==1
    assert checkpoint.read_bytes()==native
    assert len(list(tmp_path.glob('*.db')))==1  # absent optional databases not fabricated


class _Busy:
    """A connection whose WAL switch hits SQLite's deadlock-avoidance SQLITE_BUSY a given number of times."""
    def __init__(self, busy, message='database is locked'):
        self.busy, self.message, self.calls = busy, message, 0

    def execute(self, sql):
        self.calls += 1
        if self.calls <= self.busy:
            raise sqlite3.OperationalError(self.message)
        return sqlite3.connect(':memory:').execute("SELECT 'wal'")


def test_wal_switch_retries_immediate_busy_and_still_reports_other_errors():
    from storage.schema import enable_wal
    db = _Busy(2)
    assert enable_wal(db) == 'wal' and db.calls == 3
    with pytest.raises(sqlite3.OperationalError, match='disk I/O'):
        enable_wal(_Busy(1, 'disk I/O error'))
    with pytest.raises(sqlite3.OperationalError, match='locked'):
        enable_wal(_Busy(10**6), timeout=0.2)
