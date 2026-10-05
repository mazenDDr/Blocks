"""Actual Chrome graph typed-wire-insertion journey using owned isolated services."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import editor_smoke as smoke


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--chrome')
    parser.add_argument('--timeout',type=float,default=180)
    args=parser.parse_args()
    if not 1 <= args.timeout <= 600:parser.error('timeout must be 1–600 seconds')
    # Actual labelled fixtures owned by this run, never the user's ignored example data.
    fixture_dir = Path(tempfile.mkdtemp(prefix='void-insertion-fixtures-'))
    spec = importlib.util.spec_from_file_location('insertion_domain_fixtures', smoke.ROOT/'examples/make_domain_fixtures.py')
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    paths = {'vision': str(fixtures.write_vision(fixture_dir/'synthetic_shapes.npz', n=12, seed=0))}
    return smoke.run(args,journey=smoke.EDITOR/'smoke/insertion.mjs', extra_env={'VOID_INSERTION_FIXTURES': json.dumps(paths)})


if __name__=='__main__':sys.exit(main())
