"""Actual native cached runs → reviewed policy → scheduled prune in owned Chrome."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import editor_smoke as smoke
from artifact_store import ArtifactStore
from graph_core.project_io import Project, load_project, save_project
from graph_core.hashing import semantic_hash
from graph_core.schema import UiDoc
from worker.tabular_run import TabularRunConfig, run_tabular

PROJECT='SYNTHETIC_cache_retention'


def seed(root):
    store=ArtifactStore(root)
    graph=load_project(smoke.ROOT/'examples/tabular_regression.project.json').graph
    next(n for n in graph.nodes if n.id=='housing').config['path']=str(smoke.ROOT/'examples/fixtures/synthetic_housing.csv')
    save_project(Project(graph,UiDoc(positions={},description='SYNTHETIC native cache retention teaching evidence',synthetic=True)),root/f'projects/{PROJECT}.project.json')
    for rid,g in [('SYNTHETIC-cache-original',graph),('SYNTHETIC-cache-variant',graph.model_copy(deep=True))]:
        if rid.endswith('variant'):next(n for n in g.nodes if n.id=='ols').config['fit_intercept']=False
        cfg=TabularRunConfig(cache='reuse',project_id=PROJECT)
        store.create_run(rid,semantic_hash(g),cfg.model_dump())
        if run_tabular(g,cfg,store,rid)!='completed':raise RuntimeError('Actual native cache seed failed')
    rows=store.node_cache_entries(PROJECT)
    if len(rows)!=19:raise RuntimeError(f'Expected actual19native entries, got{len(rows)}')
    return store


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output');parser.add_argument('--chrome');parser.add_argument('--timeout',type=float,default=180)
    args=parser.parse_args()
    if not 1<=args.timeout<=600:parser.error('timeout must be1–600seconds')
    root=Path(tempfile.mkdtemp(prefix='void-cache-retention-source-'))
    store=seed(root)
    code=smoke.run(args,workbench=root,journey=smoke.EDITOR/'smoke/cacheRetention.mjs',fixture='SYNTHETIC regression; real native cache/prune/scheduler, no invented values')
    if code==0 and not all(store.verify(a['sha256']) for rid in ['SYNTHETIC-cache-original','SYNTHETIC-cache-variant'] for a in store.artifacts(rid)):
        raise RuntimeError('Recorded native run artifact damaged by cache retention')
    return code


if __name__=='__main__':sys.exit(main())
