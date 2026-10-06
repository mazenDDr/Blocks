"""Convert the labelled SYNTHETIC housing teaching fixture to flat JSONL."""
import json
from pathlib import Path
import pandas as pd
from tabular.core import clean

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    frame=pd.read_csv(root/'fixtures/synthetic_housing.csv')
    (root/'fixtures/synthetic_housing.jsonl').write_text(''.join(json.dumps(clean(record),ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n' for record in frame.to_dict('records')))
    graph=json.loads((root/'tabular_regression.project.json').read_text())
    source=next(n for n in graph['nodes'] if n['id']=='housing')
    source.update(type='tabular.jsonl_source',config={'path':'examples/fixtures/synthetic_housing.jsonl'})
    (root/'jsonl_regression.project.json').write_text(json.dumps(graph,indent=2)+'\n')
    ui=json.loads((root/'tabular_regression.ui.json').read_text())
    ui.update(synthetic=True,description='SYNTHETIC housing teaching data in flat JSONL. Exact byte provenance, native pandas / scikit-learn preprocessing and regression. No real-world housing benchmark.')
    (root/'jsonl_regression.ui.json').write_text(json.dumps(ui,indent=2)+'\n')
