"""Inert, bounded project packages. Import never installs or executes operations."""
from __future__ import annotations
import base64
import hashlib
from importlib.metadata import version
import json
from pathlib import Path, PurePosixPath
from pydantic import BaseModel, Field, ValidationError
from graph_core.schema import Graph,UiDoc
from graph_core.registry import get_op
from graph_core.hashing import semantic_hash
from graph_core.project_io import Project,save_project
from graph_core.validate import validate
from scale.common import IntegrationError, digest, secret_free
from tabular.core import resolve_path
from .sdk import LOADED

MAX=8*1024*1024
class Resource(BaseModel):
    sha256:str=Field(pattern=r"^[0-9a-f]{64}$")
    size:int=Field(ge=0,le=MAX)
    base64:str=Field(max_length=12*1024*1024)
class Dependency(BaseModel):
    operation:str
    version:str|None
    implementation:str|dict
class Package(BaseModel):
    model_config={"extra":"forbid"}
    format:str
    identity:str
    graph:dict
    ui:dict
    originalGraphHash:str
    dependencies:list[Dependency]=Field(max_length=256)
    resources:dict[str,Resource]
    environment:dict
    requirements:str

def build_package(graph,ui,include_csv=False):
    secret_free(graph.to_json());secret_free(ui)
    resources={}
    g=graph.model_copy(deep=True)
    if include_csv:
        for n in g.nodes:
            if n.type in ("tabular.csv_source", "tabular.jsonl_source"):
                raw=resolve_path(n.config["path"]).read_bytes()
                if sum(r["size"] for r in resources.values())+len(raw)>MAX:raise IntegrationError("package_limit","Local CSV/JSONL resources exceed 8 MiB.")
                sha=hashlib.sha256(raw).hexdigest();key="data/"+sha+(".jsonl" if n.type=="tabular.jsonl_source" else ".csv")
                resources[key]={"sha256":sha,"size":len(raw),"base64":base64.b64encode(raw).decode()}
                n.config["path"]="package://"+key
    dependencies=[]
    for typ in sorted({n.type for n in g.nodes}):
        op=get_op(typ)
        dependencies.append({"operation":typ,"version":op.version if op else None,"implementation":({k:v for k,v in LOADED[typ].items() if k in ("name","version","sha256")} if typ in LOADED else "builtin" if op else "unavailable")})
    p={"format":"void-project-package/1","graph":g.to_json(),"ui":ui,"originalGraphHash":semantic_hash(graph),"dependencies":dependencies,"resources":resources,
       "environment":{"python":"3.13","project-void":"0.0.0","numpy":version("numpy"),"scikit-learn":version("scikit-learn")},
       "requirements":"Sources outside embedded CSV/JSONL files, credentials and native environments must be configured separately. No run artifacts or credentials included."}
    p["identity"]=digest(p)
    return p

def inspect_package(p):
    try:
        Package.model_validate(p)
    except ValidationError:
        raise IntegrationError("package_invalid", "Package manifest/resources/dependency fields are invalid.") from None
    if p.get("format")!="void-project-package/1" or p.get("identity")!=digest({k:v for k,v in p.items() if k!="identity"}):raise IntegrationError("package_integrity","Package identity differs.")
    secret_free(p)
    graph=Graph.model_validate(p["graph"]);UiDoc.model_validate(p["ui"])
    if len(p["resources"])>64:raise IntegrationError("package_limit","Too many resources.")
    total=0
    for key,f in p["resources"].items():
        path=PurePosixPath(key)
        if path.is_absolute() or ".." in path.parts or not key.startswith("data/") or "\\" in key:raise IntegrationError("package_path","Unsafe package resource path.")
        data=base64.b64decode(f["base64"],validate=True);total+=len(data)
        if len(data)!=f["size"] or hashlib.sha256(data).hexdigest()!=f["sha256"]:raise IntegrationError("package_integrity","Resource bytes differ.")
        if total>MAX:raise IntegrationError("package_limit","Package resources exceed 8 MiB.")
    missing=[]
    for d in p["dependencies"]:
        op=get_op(d["operation"])
        if not op or op.version!=d["version"] or (isinstance(d["implementation"],dict) and LOADED.get(d["operation"],{}).get("sha256")!=d["implementation"]["sha256"]):missing.append(d)
    return {"identity":p["identity"],"missingDependencies":missing,"executionBlocked":bool(missing),"graphKind":graph.graphKind,"resources":len(p["resources"]),"bytes":total,"installsOrExecutesCode":False}

def import_package(p,projects,pid):
    report=inspect_package(p)
    root=Path(projects)/"packages"/p["identity"]
    root.mkdir(parents=True,exist_ok=True)
    for key,f in p["resources"].items():
        dest=root/key
        if root.is_symlink() or root.resolve() not in dest.resolve().parents:
            raise IntegrationError("package_path", "Package destination escapes its owned resource directory.")
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(base64.b64decode(f["base64"]))
    graph=Graph.model_validate(p["graph"])
    for n in graph.nodes:
        path=n.config.get("path","")
        if isinstance(path,str) and path.startswith("package://"):
            key=path[10:]
            if key not in p["resources"]:raise IntegrationError("package_resource_missing","Source refers to an absent resource.")
            n.config["path"]=str((root/key).resolve())
    graph.__pydantic_extra__={**(graph.__pydantic_extra__ or {}),"packageIdentity":p["identity"],"packageDependencies":p["dependencies"]}
    save_project(Project(graph,UiDoc.model_validate(p["ui"])),Path(projects)/(pid+".project.json"))
    return {**report,"projectId":pid,"graph":graph.to_json(),"ui":p["ui"]}
