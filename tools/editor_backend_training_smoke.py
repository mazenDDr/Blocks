"""Actual Chrome JAX training run using owned native services, on a freshly generated SYNTHETIC shapes folder."""
import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

import editor_smoke as smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--chrome')
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 600:
        parser.error('timeout must be 1–600 seconds')
    # examples/data/ is generated and gitignored, so the journey must not depend on a local copy.
    spec = importlib.util.spec_from_file_location("make_shapes10", smoke.ROOT / "examples" / "make_shapes10.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    data = Path(tempfile.mkdtemp(prefix="void-shapes-")) / "shapes10"
    mod.generate(data, per_class=6, seed=0)
    return smoke.run(args, journey=smoke.EDITOR / 'smoke/backendTraining.mjs', extra_env={"VOID_SHAPES_DIR": str(data)})


if __name__ == '__main__':
    sys.exit(main())
