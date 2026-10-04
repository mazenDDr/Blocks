from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from pydantic import BaseModel, Field
from scale.common import IntegrationError, secret_free

class Manifest(BaseModel):
    model_config={"extra":"forbid"}
    schemaVersion:str=Field("1.0.0",pattern=r"^1\.0\.0$")
    name:str=Field(pattern=r"^[a-z][a-z0-9_-]{2,63}$")
    version:str=Field(pattern=r"^\d+\.\d+\.\d+$")
    operation:str=Field(pattern=r"^community\.[a-z][a-z0-9_.]+$")
    implementation:str=Field(pattern=r"^[a-zA-Z0-9_-]+\.py$")
    className:str=Field(pattern=r"^[A-Za-z][A-Za-z0-9_]+$")
    sha256:str=Field(pattern=r"^[0-9a-f]{64}$")
    inputs:dict[str,str]
    outputs:dict[str,str]
    settingsSchema:dict
    state:str=Field("none",pattern=r"^none$")
    effects:list[str]=Field(default_factory=list,max_length=0)
    inspector:dict
    documentation:str=Field(min_length=10)
    exampleGraph:dict
    cases:list[dict]=Field(min_length=1,max_length=16)
    nativeDependencies:dict[str,str]

LOADED={}
def read_manifest(path):
    p=Path(path).resolve()
    if p.stat().st_size>1024*1024:raise IntegrationError("plugin_manifest_limit","Manifest exceeds 1 MiB.")
    m=Manifest.model_validate(json.loads(p.read_text()))
    src=p.parent/m.implementation
    if src.is_symlink() or not src.is_file() or hashlib.sha256(src.read_bytes()).hexdigest()!=m.sha256:
        raise IntegrationError("plugin_integrity","Implementation hash differs or source is a symlink.")
    secret_free(m.model_dump())
    return m,src

def load_trusted(path):
    from graph_core import registry
    from tabular.core import TabularOperation
    from importlib.metadata import version
    m,src=read_manifest(path)
    for dep,wanted in m.nativeDependencies.items():
        if version(dep)!=wanted:raise IntegrationError("plugin_dependency",f"{dep} must be {wanted}")
    if m.operation in LOADED:
        if LOADED[m.operation]["sha256"]!=m.sha256 or LOADED[m.operation]["version"]!=m.version:raise IntegrationError("plugin_duplicate","A different package already provides this operation.")
        return registry.get_op(m.operation)
    # This executes explicitly trusted native code. It is not an untrusted Python sandbox.
    spec=importlib.util.spec_from_file_location("void_plugin_"+m.sha256,src)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    cls=getattr(mod,m.className)
    if not issubclass(cls,TabularOperation) or cls.type!=m.operation or cls.version!=m.version or cls.graph_kind!="tabular":
        raise IntegrationError("plugin_contract","SDK currently supports pure tabular operations with exact type/version.")
    op=cls()
    op.Config()  # Registry-generated forms require usable defaults.
    if dict(op.in_kinds)!=m.inputs or dict(op.out_kinds)!=m.outputs or tuple(op.inputs)!=tuple(m.inputs) or tuple(op.outputs)!=tuple(m.outputs) or op.Config.model_json_schema()!=m.settingsSchema:
        raise IntegrationError("plugin_contract","Implementation ports/settings differ from the manifest.")
    registry.register(cls)
    LOADED[m.operation]={"name":m.name,"version":m.version,"sha256":m.sha256,"manifest":str(Path(path).resolve()),"inspector":m.inspector,"documentation":m.documentation}
    return op

def activate_environment():
    for path in filter(None,os.environ.get("VOID_PLUGIN_MANIFESTS","").split(os.pathsep)):
        load_trusted(path)

def conformance(path):
    import pandas as pd
    from tabular.core import Table,ExecCtx,schema_of,table_type
    from graph_core.schema import Graph
    from graph_core.validate import require_executable
    m,_=read_manifest(path);op=load_trusted(path)
    outcomes=[]
    for case in m.cases:
        cfg=op.Config.model_validate(case["config"])
        ins={k:Table(pd.DataFrame(v),partition=case.get("partition","train")) for k,v in case["inputs"].items()}
        before={k:v.df.copy(deep=True) for k,v in ins.items()}
        types={k:table_type(schema_of(v.df),len(v.df),v.partition) for k,v in ins.items()}
        inferred=op.infer(cfg,types,"fixture")
        outs,summary=op.execute(cfg,ins,ExecCtx("fixture"))
        repeated,_=op.execute(cfg,ins,ExecCtx("fixture"))
        if set(outs)!=set(m.outputs) or set(inferred)!=set(m.outputs):raise IntegrationError("plugin_conformance","Output ports differ.")
        for k,wanted in case["expected"].items():
            expected=pd.DataFrame(wanted)
            pd.testing.assert_frame_equal(outs[k].df.reset_index(drop=True),expected,check_dtype=False,rtol=1e-12,atol=1e-12)
            pd.testing.assert_frame_equal(outs[k].df,repeated[k].df)
            if schema_of(outs[k].df)!=inferred[k].columns:raise IntegrationError("plugin_conformance","Inferred schema differs from native output.")
        for k,v in ins.items():pd.testing.assert_frame_equal(v.df,before[k])
        explanation=op.explain(cfg,types,inferred)
        if not explanation or not summary:raise IntegrationError("plugin_conformance","Explanation and inspection summary are required.")
        json.dumps(summary,allow_nan=False)
        outcomes.append({"name":case["name"],"passed":True})
    require_executable(Graph.model_validate(m.exampleGraph))
    return {"operation":m.operation,"version":m.version,"sha256":m.sha256,"cases":outcomes,"passed":True,"scope":"declared numerical fixtures, schema, input immutability and example graph; native code is trusted"}

if __name__=="__main__":
    print(json.dumps(conformance(sys.argv[1]),indent=2))
