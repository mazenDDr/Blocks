"""Owned recovery-fixture legacy headers and exact pre-browser native-row checks."""
from contextlib import closing
import hashlib
import json
import sqlite3
from storage.schema import definitions, guard, inspect, statements


def census(root):
    result = {}
    for relative, (kind, schema) in definitions().items():
        path = root / relative
        if not path.is_file():
            continue
        with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
            version = guard(db, kind, (statements(schema),))
            tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            values = {name:db.execute(f'SELECT * FROM "{name}" ORDER BY rowid').fetchall() for name in tables}
            digest = hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        result[relative] = {'version':version,'rowsSha256':digest}
    return result


def legacy_fixture(root):
    """Reset only application header stamps of our stopped SYNTHETIC native fixture.

    Rows/schema/CAS are actual native seed outputs. This is an explicitly constructed
    compatible version0 fixture, not evidence of opening arbitrary historical code.
    """
    before = census(root)
    for relative, (kind, schema) in definitions().items():
        path = root / relative
        if path.is_file():
            with closing(sqlite3.connect(path)) as db:
                guard(db, kind, (statements(schema),))
                db.execute('PRAGMA user_version=0');db.execute('PRAGMA application_id=0');db.commit()
    after = census(root)
    assert all(row['version']==0 and row['rowsSha256']==before[name]['rowsSha256'] for name,row in after.items())
    return {'fixture':'SYNTHETIC native data with explicitly constructed legacy version0 headers','before':before,'legacy':after}


def check_ready(root, legacy):
    current = census(root)
    expected = legacy['legacy']
    assert set(current)==set(expected)
    assert all(row['version']==1 and row['rowsSha256']==expected[name]['rowsSha256'] for name,row in current.items()), 'Startup migration altered native rows or left legacy metadata'
    return {'restored':current,'inventory':inspect(root),'verifiedBeforeBrowserActions':True}
