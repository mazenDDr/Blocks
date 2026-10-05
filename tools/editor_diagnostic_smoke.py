"""Actual Chrome native nested-diagnostic navigation on labelled SYNTHETIC contracts."""
import argparse
import importlib.util
import json
import sys
import editor_smoke as smoke


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--chrome')
    parser.add_argument('--timeout',type=float,default=180)
    args=parser.parse_args()
    if not 1 <= args.timeout <= 600:parser.error('timeout must be 1–600 seconds')
    spec=importlib.util.spec_from_file_location('diagnostic_fixture',smoke.ROOT/'examples/make_diagnostic_fixture.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    graphs={kind:fixture.diagnostic_graph(kind).to_json() for kind in ['nested','repeat','select']}
    return smoke.run(args,journey=smoke.EDITOR/'smoke/diagnostics.mjs',extra_env={'VOID_DIAGNOSTIC_FIXTURES':json.dumps(graphs)})


if __name__=='__main__':sys.exit(main())
