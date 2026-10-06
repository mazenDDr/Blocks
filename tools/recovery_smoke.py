"""Actual Chrome baseline → offline backup → source deletion → restored editor.

Only fresh isolated workbenches created by this runner are removed. Native learned
tabular/vision/NLP/speech recovery is independently exercised by pytest.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
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
        if args.json_agent:
            import editor_json_agent_smoke as json_smoke
            json_smoke.available()
        seed = argparse.Namespace(output=str(out / "seed"), chrome=args.chrome, timeout=args.timeout)
        if smoke.run(seed):
            raise RuntimeError("Initial browser baseline failed; no backup taken.")
        source = out / "seed/workbench"
        extra_env = {"VOID_RECOVERY_SEED": str(out / "seed/evidence.json")}
        if args.json_agent:
            json_seed = argparse.Namespace(output=str(out / "json-seed"), chrome=args.chrome, timeout=args.timeout)
            if smoke.run(json_seed, workbench=source, journey=smoke.EDITOR / "smoke/jsonAgent.mjs", fixture=json_smoke.FIXTURE):
                raise RuntimeError("Real Ollama JSON source browser failed; no backup taken.")
            extra_env["VOID_JSON_RECOVERY_SEED"] = str(out / "json-seed/evidence.json")
            result["jsonAgent"] = {"provider": "actual installed Ollama qwen3.5:2b", "seed": "json-seed/evidence.json"}
        if args.trackers:
            seeded = subprocess.run([sys.executable, str(smoke.ROOT / "tools/tracker_recovery_seed.py"), "--workbench", str(source)],
                                    env=smoke.isolated_env(), capture_output=True, text=True, timeout=240)
            if seeded.returncode:
                raise RuntimeError("Native tracker seed failed: " + seeded.stderr[-2000:])
            tracker_evidence = out / "trackers.json"
            tracker_evidence.write_text(json.dumps(json.loads(seeded.stdout.splitlines()[-1]), indent=2) + "\n")
            extra_env["VOID_RECOVERY_TRACKERS"] = str(tracker_evidence)
        result["backup"] = create(source, out / "backup", offline=True,
                                  links="internal" if args.trackers else "reject", omit_wandb_external_logs=args.trackers)
        sha = result["backup"]["manifestSha256"]
        verify(out / "backup", manifest_sha256=sha)
        shutil.rmtree(source)  # own disposable generated workbench only, after verified backup
        result["sourceDeleted"] = not source.exists()
        result["restore"] = restore(out / "backup", out / "recovered", trusted=True, manifest_sha256=sha)
        check = argparse.Namespace(output=str(out / "check"), chrome=args.chrome, timeout=args.timeout)
        if smoke.run(check, workbench=out / "recovered", journey=smoke.EDITOR / "smoke/recovery.mjs",
                     extra_env=extra_env):
            raise RuntimeError("Restored browser recovery failed.")
        if args.json_agent:
            json_check = argparse.Namespace(output=str(out / "json-check"), chrome=args.chrome, timeout=args.timeout)
            if smoke.run(json_check, workbench=out / "recovered", journey=smoke.EDITOR / "smoke/jsonAgent.mjs",
                         extra_env=extra_env, fixture=json_smoke.FIXTURE):
                raise RuntimeError("Restored native JSON browser/provider execution failed.")
            result["jsonAgent"]["restored"] = "json-check/evidence.json"
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
    parser.add_argument("--trackers", action="store_true", help="Seed real local MLflow/offline W&B; use v2 links with explicit external diagnostic-log omission.")
    parser.add_argument("--json-agent", action="store_true", help="Also seed/recover a pinned JSON version and invoke real installed local Ollama; fails without the provider, no fixture substitution.")
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
