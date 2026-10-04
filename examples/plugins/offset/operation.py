"""Trusted SDK example: a declared constant shift, no fitted state."""
from pydantic import BaseModel, Field
from tabular.core import TabularOperation,need_columns,need_numeric

class Settings(BaseModel):
    model_config={"extra":"forbid"}
    column:str="signal"
    offset:float=Field(1.0,allow_inf_nan=False)

class Offset(TabularOperation):
    type="community.table_offset"
    version="1.0.0"
    inputs=("table",);outputs=("table",)
    in_kinds={"table":"table"};out_kinds={"table":"table"}
    Config=Settings
    def infer(self,cfg,ins,node_id):
        need_columns(ins["table"],[cfg.column],"table","offset")
        need_numeric(ins["table"],[cfg.column],"table","offset")
        import copy
        t=copy.deepcopy(ins["table"])
        for c in t.info["columns"]:
            if c["name"]==cfg.column:c["dtype"]="float"
        return {"table":t}
    def execute(self,cfg,ins,ctx):
        t=ins["table"]
        out=t.df.copy(deep=True)
        out[cfg.column]=out[cfg.column].astype(float)+cfg.offset
        return {"table":t.derive(out,operation=self.type)}, {"column":cfg.column,"offset":cfg.offset,"rows":len(out),"equation":"output = input + declared offset","fitted":False}
    def explain(self,cfg,ins,outs):
        return {"equation":"output = input + declared offset","state":"none","effects":[],"offset":cfg.offset}
