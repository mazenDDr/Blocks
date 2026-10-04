"""Milestone 5 read-only route for the unsupervised views: bounded 2-D points from recorded tables, optionally coloured by a column of another recorded table
(joined on the source row id). Nothing is recomputed."""
from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from .tabular_inspection import TabularRun

MAX_POINTS = 5000


def _bad(status: int, message: str) -> HTTPException:
    return HTTPException(status, {"code": "inspect_invalid", "message": message})


def register(app: FastAPI, sv) -> None:
    store = sv.store

    def read(tr: TabularRun, node: str, port: str) -> pd.DataFrame:
        a = tr.outputs.get((node, port))
        if a is None or a["meta"].get("valueKind") != "table":
            raise _bad(404, f"run {tr.run_id} recorded no table at {node}.{port}")
        return pd.read_csv(store.path_of(a["sha256"]), index_col="row_id")

    @app.get("/api/runs/{rid}/points")
    def points(rid: str, node: str, port: str, x: str, y: str, color_node: str | None = None, color_port: str | None = None, color: str | None = None, limit: int = Query(MAX_POINTS, ge=1, le=MAX_POINTS)):
        tr = TabularRun(store, rid)
        df = read(tr, node, port)
        for c in (x, y):
            if c not in df.columns:
                raise _bad(422, f"column '{c}' is not in {node}.{port} (columns: {list(df.columns)})")
        col = None
        src = f"{node}.{port}"
        if color:
            if color_node:
                cdf = read(tr, color_node, color_port or "assignments")
                src = f"{color_node}.{color_port or 'assignments'}"
            else:
                cdf = df
            if color not in cdf.columns:
                raise _bad(422, f"colour column '{color}' is not in {src} (columns: {list(cdf.columns)})")
            col = cdf[color].reindex(df.index)
        total = len(df)
        step = max(1, -(-total // limit))
        sel = df.iloc[::step]
        pts = [{"id": int(i), "x": float(r[x]), "y": float(r[y]), "c": (None if col is None or pd.isna(col.loc[i]) else (col.loc[i].item() if hasattr(col.loc[i], "item") else col.loc[i]))} for i, r in sel.iterrows()]
        return {"runId": rid, "node": node, "port": port, "x": x, "y": y, "colorBy": {"column": color, "from": src} if color else None, "points": pts, "total": total, "shown": len(pts),
                "downsampled": step > 1, "provenance": {"runId": rid, "graphHash": tr.graph_hash, "source": "recorded table; every k-th row when downsampled"}}
