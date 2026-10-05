"""Actual Chrome graph measured-arrangement journey using owned isolated services."""
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
    fixture_dir = Path(tempfile.mkdtemp(prefix='void-arrangement-fixtures-'))
    spec = importlib.util.spec_from_file_location('arrangement_domain_fixtures', smoke.ROOT/'examples/make_domain_fixtures.py')
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    paths = {
        'vision_segmentation_synthetic': str(fixtures.write_vision(fixture_dir/'synthetic_shapes.npz', n=12, seed=0)),
        'nlp_token_classification': str(fixtures.write_ner(fixture_dir/'synthetic_ner.jsonl', n=24, seed=0)),
        'speech_ctc_tones': str(fixtures.write_audio(fixture_dir/'synthetic_tones.npz', n=12, seed=0)),
    }
    return smoke.run(args,journey=smoke.EDITOR/'smoke/arrangement.mjs', extra_env={'VOID_ARRANGEMENT_FIXTURES': json.dumps(paths)})


if __name__=='__main__':sys.exit(main())
