"""Recorded runs and revisioned user-authored research metadata."""
from fastapi import Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from research.records import Records, RecordError


class Annotation(BaseModel):
    model_config={'extra':'forbid'}
    mutationId:str=Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')
    expectedRevision:int=Field(ge=0,strict=True)
    expectedGraphHash:str=Field(pattern=r'^[0-9a-f]{64}$')
    author:str=Field(min_length=1,max_length=100)
    note:str=Field(max_length=5000)
    tags:list[str]=Field(max_length=20)
    @field_validator('author')
    @classmethod
    def author_present(cls,value):
        if not value.strip():raise ValueError('Declare an author label.')
        return value
    @field_validator('tags')
    @classmethod
    def bounded_tags(cls,values):
        if any(not isinstance(v,str) or not v.strip() or v!=v.strip() or len(v)>60 for v in values):raise ValueError('Tags must be nonempty trimmed strings of1–60 characters.')
        if len(set(values))!=len(values):raise ValueError('Duplicate tags are not admitted.')
        return values


def register(app,sv):
    records=Records(sv.store)
    @app.exception_handler(RecordError)
    async def error(_request:Request,e:RecordError):
        return JSONResponse({'detail':{'code':e.code,'message':e.message}},status_code=e.status)
    @app.get('/api/research/runs')
    def catalogue(query:str=Query('',max_length=200),tag:str|None=Query(None,min_length=1,max_length=60),
                  projectId:str|None=Query(None,min_length=1,max_length=64),kind:str|None=Query(None,pattern=r'^(model|tabular|agent|rl|procedure|unsup|domain)$'),
                  status:str|None=Query(None,pattern=r'^(queued|preparing|running|paused|cancelling|completed|failed|cancelled)$'),
                  after:str=Query('',max_length=128),limit:int=Query(25,ge=1,le=100)):
        return records.catalogue(query=query,tag=tag,project_id=projectId,kind=kind,status=status,after=after,limit=limit)
    @app.get('/api/research/runs/{run_id}')
    def inspect(run_id:str):return records.get(run_id)
    @app.put('/api/research/runs/{run_id}/annotation')
    def annotate(run_id:str,body:Annotation):return records.update(run_id,body.model_dump())
    @app.get('/api/research/runs/{run_id}/annotation/history')
    def history(run_id:str,after:int=Query(0,ge=0),limit:int=Query(25,ge=1,le=100)):
        return records.history(run_id,after,limit)
