"""Actual Chrome baseline → offline backup → source deletion → restored editor.

Only fresh isolated workbenches created by this runner are removed. Native learned
tabular/vision/NLP/speech recovery is independently exercised by pytest.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

import editor_smoke as smoke
from workbench_backup.core import create, restore, verify


def run(args):
    out = Path(args.output).expanduser().resolve() if args.output else Path(tempfile.mkdtemp(prefix="void-recovery-smoke-"))
    if out.is_relative_to(smoke.ROOT):
        raise ValueError("Evidence must be outside the repository.")
    if args.output:
        out.mkdir(parents=True, exist_ok=False)
    result = {"status": "failed", "fixture": "SYNTHETIC native state recovery", "evidenceDirectory": str(out)}
    try:
        seed = argparse.Namespace(output=str(out / "seed"), chrome=args.chrome, timeout=args.timeout)
        if smoke.run(seed):
            raise RuntimeError("Initial browser baseline failed; no backup taken.")
        source = out / "seed/workbench"
        result["backup"] = create(source, out / "backup", offline=True)
        sha = result["backup"]["manifestSha256"]
        verify(out / "backup", manifest_sha256=sha)
        shutil.rmtree(source)  # own disposable generated workbench only, after verified backup
        result["sourceDeleted"] = not source.exists()
        result["restore"] = restore(out / "backup", out / "recovered", trusted=True, manifest_sha256=sha)
        check = argparse.Namespace(output=str(out / "check"), chrome=args.chrome, timeout=args.timeout)
        if smoke.run(check, workbench=out / "recovered", journey=smoke.EDITOR / "smoke/recovery.mjs",
                     extra_env={"VOID_RECOVERY_SEED": str(out / "seed/evidence.json")}):
            raise RuntimeError("Restored browser recovery failed.")
        result["status"] = "passed"
    except Exception as error:
        result["error"] = str(error)
    finally:
        (out / "recovery.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="new evidence directory outside the repository")
    parser.add_argument("--chrome")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 600:
        parser.error("--timeout must be 1–600 seconds")
    try:
        return run(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Recovery setup failed: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
