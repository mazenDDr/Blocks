"""Bounded run annotations/catalogue; native model/store source identities unchanged."""
from contextlib import closing
import hashlib
import json
import math
import sqlite3
import time

SCHEMA='''
CREATE TABLE IF NOT EXISTS annotations(run_id TEXT PRIMARY KEY, graph_hash TEXT NOT NULL, revision INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS revisions(run_id TEXT NOT NULL, revision INTEGER NOT NULL, author TEXT NOT NULL, note TEXT NOT NULL, tags TEXT NOT NULL, recorded_at REAL NOT NULL, PRIMARY KEY(run_id,revision));
CREATE TABLE IF NOT EXISTS mutations(id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, result TEXT NOT NULL);
'''
POLICY='User-authored research metadata, not measured results. Shared-token holders can edit; author is caller-declared, not authenticated ownership. Clearing current metadata preserves earlier revisions. Catalogue pages are current reads, not a frozen snapshot.'


class RecordError(Exception):
    def __init__(self,code,message,status=409):self.code,self.message,self.status=code,message,status


class Records:
    def __init__(self,artifacts):
        self.artifacts=artifacts
        self.path=artifacts.root/'research.sqlite'
        with closing(self.db()) as db:db.executescript(SCHEMA)

    def db(self):
        db=sqlite3.connect(self.path,timeout=10,uri=True)
        db.row_factory=sqlite3.Row
        return db

    def run(self,run_id):
        run=self.artifacts.get_run(run_id)
        if run is None:raise RecordError('E_RECORD_RUN','Recorded run does not exist.',404)
        return run

    @staticmethod
    def metadata(run):
        return {'runId':run['id'],'graphHash':run['graph_hash'],'projectId':run['config'].get('project_id'),
                'kind':run['config'].get('kind','model'),'status':run['status'],'createdAt':run['created_at'],'updatedAt':run['updated_at']}

    @staticmethod
    def annotation(row):
        if row is None:return {'revision':0,'note':'','tags':[],'author':None,'recordedAt':None}
        try:tags=json.loads(row['tags'])
        except (TypeError,ValueError):raise RecordError('E_RECORD_INTEGRITY','Stored annotation tags are malformed.') from None
        if (type(row['revision']) is not int or row['revision']<1 or not isinstance(row['note'],str) or len(row['note'])>5000
                or not isinstance(row['author'],str) or not row['author'].strip() or len(row['author'])>100
                or not isinstance(tags,list) or len(tags)>20 or any(not isinstance(t,str) or not t.strip() or t!=t.strip() or len(t)>60 for t in tags)
                or len(set(tags))!=len(tags) or not isinstance(row['recorded_at'],(int,float)) or not math.isfinite(row['recorded_at']) or row['recorded_at']<=0):
            raise RecordError('E_RECORD_INTEGRITY','Stored annotation violates its bounded metadata contract.')
        return {'revision':row['revision'],'note':row['note'],'tags':tags, 'author':row['author'],'recordedAt':row['recorded_at']}

    def get(self,run_id):
        run=self.run(run_id)
        with closing(self.db()) as db:
            row=db.execute('SELECT r.*,a.graph_hash FROM annotations a LEFT JOIN revisions r ON r.run_id=a.run_id AND r.revision=a.revision WHERE a.run_id=?',(run_id,)).fetchone()
        if row is not None and row['graph_hash']!=run['graph_hash']:raise RecordError('E_RECORD_IDENTITY','Annotation graph identity differs from the recorded run.')
        return {**self.metadata(run),'annotation':self.annotation(row),'policy':POLICY}

    def update(self,run_id,body):
        run=self.run(run_id)
        if body['expectedGraphHash']!=run['graph_hash']:raise RecordError('E_RECORD_IDENTITY','Reviewed run graph identity changed; reload before annotating.')
        data={'runId':run_id,'graphHash':run['graph_hash'],**body}
        fingerprint=hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        with closing(self.db()) as db,db:
            db.execute('BEGIN IMMEDIATE')
            duplicate=db.execute('SELECT fingerprint,result FROM mutations WHERE id=?',(body['mutationId'],)).fetchone()
            if duplicate:
                if duplicate['fingerprint']!=fingerprint:raise RecordError('E_RECORD_MUTATION','Mutation ID belongs to different reviewed values.')
                return {**json.loads(duplicate['result']),'idempotentReplay':True}
            current=db.execute('SELECT graph_hash,revision FROM annotations WHERE run_id=?',(run_id,)).fetchone()
            if current and current['graph_hash']!=run['graph_hash']:raise RecordError('E_RECORD_IDENTITY','Run graph identity changed; no annotation replaced.')
            if current:
                previous=db.execute('SELECT * FROM revisions WHERE run_id=? AND revision=?',(run_id,current['revision'])).fetchone()
                if previous is None:raise RecordError('E_RECORD_INTEGRITY','Current annotation revision is missing; no metadata replaced.')
                self.annotation(previous)
            revision=current['revision'] if current else 0
            if revision!=body['expectedRevision']:raise RecordError('E_RECORD_CONFLICT','Annotation changed; reload and review it before saving.')
            recorded=time.time()
            db.execute('INSERT INTO revisions VALUES(?,?,?,?,?,?)',(run_id,revision+1,body['author'],body['note'],json.dumps(body['tags'],ensure_ascii=False),recorded))
            db.execute('INSERT INTO annotations VALUES(?,?,?) ON CONFLICT(run_id) DO UPDATE SET revision=excluded.revision',(run_id,run['graph_hash'],revision+1))
            result={**self.metadata(run),'mutationId':body['mutationId'],'annotation':{'revision':revision+1,'note':body['note'],'tags':body['tags'],'author':body['author'],'recordedAt':recorded},'policy':POLICY}
            db.execute('INSERT INTO mutations VALUES(?,?,?)',(body['mutationId'],fingerprint,json.dumps(result,ensure_ascii=False)))
        return {**result,'idempotentReplay':False}

    def history(self,run_id,after=0,limit=25):
        self.get(run_id) # verifies run/current graph identity
        with closing(self.db()) as db:
            rows=db.execute('SELECT * FROM revisions WHERE run_id=? AND revision>? ORDER BY revision LIMIT ?',(run_id,after,limit+1)).fetchall()
        values=[self.annotation(row) for row in rows[:limit]]
        return {'runId':run_id,'revisions':values,'nextAfter':values[-1]['revision'] if len(rows)>limit else None,'policy':POLICY}

    def catalogue(self,*,query='',tag=None,project_id=None,kind=None,status=None,after='',limit=25):
        # Only recorded run metadata and bounded authored text; never load artifact bytes/native providers.
        with closing(self.db()) as db:
            db.execute('ATTACH DATABASE ? AS execution',(self.artifacts.db_path.resolve().as_uri()+'?mode=ro',))
            db.create_function('contains_folded',2,lambda haystack,needle: str(needle).casefold() in str(haystack or '').casefold(),deterministic=True)
            rows=db.execute('''SELECT x.id,x.graph_hash,x.status,x.created_at,x.updated_at,json_extract(x.config,'$.project_id') project_id,
                coalesce(json_extract(x.config,'$.kind'),'model') kind,r.revision,r.author,r.note,r.tags,r.recorded_at,a.graph_hash annotation_hash
                FROM execution.runs x LEFT JOIN annotations a ON a.run_id=x.id
                LEFT JOIN revisions r ON r.run_id=a.run_id AND r.revision=a.revision
                WHERE x.id>? AND (? IS NULL OR json_extract(x.config,'$.project_id')=?)
                AND (? IS NULL OR coalesce(json_extract(x.config,'$.kind'),'model')=?) AND (? IS NULL OR x.status=?)
                AND (? IS NULL OR EXISTS(SELECT 1 FROM json_each(coalesce(r.tags,'[]')) WHERE value=?))
                AND contains_folded(x.id||' '||x.graph_hash||' '||coalesce(json_extract(x.config,'$.project_id'),'')||' '||coalesce(r.note,'')||' '||coalesce(r.tags,''),?)
                ORDER BY x.id LIMIT ?''',(after,project_id,project_id,kind,kind,status,status,tag,tag,query,limit+1)).fetchall()
        values=[]
        for row in rows[:limit]:
            if row['annotation_hash'] is not None and row['annotation_hash']!=row['graph_hash']:raise RecordError('E_RECORD_IDENTITY','Annotation graph identity differs; no artifact loaded.')
            if row['annotation_hash'] is not None and row['revision'] is None:raise RecordError('E_RECORD_INTEGRITY','Current annotation revision is missing.')
            values.append({'runId':row['id'],'graphHash':row['graph_hash'],'projectId':row['project_id'],'kind':row['kind'],'status':row['status'],
                           'createdAt':row['created_at'],'updatedAt':row['updated_at'],'annotation':self.annotation(row if row['revision'] is not None else None)})
        return {'runs':values,'nextAfter':values[-1]['runId'] if len(rows)>limit else None,'query':query,'tag':tag,'projectId':project_id,'kind':kind,'status':status,'policy':POLICY}
