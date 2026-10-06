"""Actual Chrome baseline → offline backup → source deletion → restored editor.

Only fresh isolated workbenches created by this runner are removed. Native learned
tabular/vision/NLP/speech recovery is independently exercised by pytest.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import editor_smoke as smoke
from workbench_backup.core import create, restore, verify


def collect_garbage(source):
    """Inject one old SYNTHETIC orphan, then collect with no grace period before backup.

    Zero grace is the most aggressive setting: every unreferenced blob of the real seeded
    workbench is deleted, so the restored journeys prove no needed blob was collected.
    """
    from maintenance.cas_gc import collect
    from artifact_store import ArtifactStore
    store = ArtifactStore(source)
    orphan = store.put_bytes(b"SYNTHETIC unreferenced CAS blob for offline collection")
    os.utime(store.path_of(orphan), (time.time() - 7 * 86400,) * 2)
    preview = collect(source, grace_seconds=0)
    if orphan not in preview["list"]:
        raise RuntimeError("Collection preview did not report the injected orphan.")
    applied = collect(source, apply=True, offline=True, grace_seconds=0)
    if applied["listSha256"] != preview["listSha256"] or store.path_of(orphan).exists():
        raise RuntimeError("Applied collection differs from its preview or kept the orphan.")
    after = collect(source, grace_seconds=0)
    if after["candidates"] or after["stalePartialFiles"]:
        raise RuntimeError("Collection left unreferenced candidates behind.")
    return {"orphan": orphan, "preview": {k: preview[k] for k in ("blobs", "referenced", "candidates", "candidateBytes", "stalePartialFiles", "databases", "scannedFiles")},
            "deleted": applied["list"], "listSha256": applied["listSha256"], "remainingBlobs": after["blobs"]}


def run(args):
    out = Path(args.output).expanduser().resolve() if args.output else Path(tempfile.mkdtemp(prefix="void-recovery-smoke-"))
    if out.is_relative_to(smoke.ROOT):
        raise ValueError("Evidence must be outside the repository.")
    if args.output:
        out.mkdir(parents=True, exist_ok=False)
    result = {"status": "failed", "fixture": "SYNTHETIC native state recovery", "evidenceDirectory": str(out)}
    try:
        if args.cache_retention:
            import editor_cache_retention_smoke as cache_smoke
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
        if args.cache_retention:
            cache_smoke.seed(source)
            cache_seed = argparse.Namespace(output=str(out / "cache-seed"), chrome=args.chrome, timeout=args.timeout)
            if smoke.run(cache_seed, workbench=source, journey=smoke.EDITOR / "smoke/cacheRetention.mjs",
                         fixture="SYNTHETIC regression; actual native scheduled cache retention"):
                raise RuntimeError("Native cache-retention browser seed failed; no backup taken.")
            extra_env["VOID_CACHE_RECOVERY_SEED"] = str(out / "cache-seed/evidence.json")
            result["cacheRetention"] = {"seed": "cache-seed/evidence.json"}
        if args.trackers:
            seeded = subprocess.run([sys.executable, str(smoke.ROOT / "tools/tracker_recovery_seed.py"), "--workbench", str(source)],
                                    env=smoke.isolated_env(), capture_output=True, text=True, timeout=240)
            if seeded.returncode:
                raise RuntimeError("Native tracker seed failed: " + seeded.stderr[-2000:])
            tracker_evidence = out / "trackers.json"
            tracker_evidence.write_text(json.dumps(json.loads(seeded.stdout.splitlines()[-1]), indent=2) + "\n")
            extra_env["VOID_RECOVERY_TRACKERS"] = str(tracker_evidence)
        if args.gc:
            result["gc"] = collect_garbage(source)
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
        if args.cache_retention:
            cache_check = argparse.Namespace(output=str(out / "cache-check"), chrome=args.chrome, timeout=args.timeout)
            if smoke.run(cache_check, workbench=out / "recovered", journey=smoke.EDITOR / "smoke/cacheRetention.mjs",
                         extra_env=extra_env, fixture="SYNTHETIC native cache policy/receipt recovery"):
                raise RuntimeError("Restored native cache policy/receipt browser failed.")
            result["cacheRetention"]["restored"] = "cache-check/evidence.json"
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
    parser.add_argument("--cache-retention", action="store_true", help="Also seed actual native cached regression/policy and verify policy/receipts/cache/run artifacts after source deletion.")
    parser.add_argument("--json-agent", action="store_true", help="Also seed/recover a pinned JSON version and invoke real installed local Ollama; fails without the provider, no fixture substitution.")
    parser.add_argument("--gc", action="store_true", help="Inject an old orphan and run offline zero-grace CAS collection on the seeded workbench before backup.")
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
