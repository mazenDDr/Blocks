"""Durable integration intentions and exact confirmed mappings, one owning process."""
import json
from storage.schema import INTEGRATIONS, open_database
import threading
import time
from contextlib import closing
from pathlib import Path
from .common import encoded

class State:
    def __init__(self, root):
        self.path = Path(root)/"integrations.sqlite"
        self.lock = threading.RLock()
        with self.lock, closing(open_database(self.path,"integrations",INTEGRATIONS)) as db, db:
            pass  # migrated transactionally before application access
    def get(self, kind, id):
        with self.lock, closing(open_database(self.path,"integrations",INTEGRATIONS)) as db:
            r = db.execute("SELECT value FROM records WHERE kind=? AND id=?",(kind,id)).fetchone()
            return json.loads(r[0]) if r else None
    def put(self, kind, id, value):
        with self.lock, closing(open_database(self.path,"integrations",INTEGRATIONS)) as db, db:
            db.execute("INSERT OR REPLACE INTO records VALUES(?,?,?,?)",(kind,id,encoded(value).decode(),time.time()))
        return value
    def all(self, kind):
        with self.lock, closing(open_database(self.path,"integrations",INTEGRATIONS)) as db:
            return [json.loads(r[0]) for r in db.execute("SELECT value FROM records WHERE kind=? ORDER BY updated DESC",(kind,))]
