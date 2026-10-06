"""Container entrypoint refusals (ADR 0074): no unauthenticated control API, TLS needs both readable files."""
import os
import subprocess
from pathlib import Path

ENTRY = Path(__file__).resolve().parents[1] / "deploy" / "entrypoint.sh"


def run(env):
    clean = {k: v for k, v in os.environ.items() if not k.startswith("VOID_")}
    return subprocess.run(["sh", str(ENTRY)], env={**clean, **env, "PATH": "/nonexistent:" + clean.get("PATH", "")}, capture_output=True, text=True, timeout=30)


def test_refuses_without_authentication():
    out = run({})
    assert out.returncode == 64 and "E_DEPLOY_AUTH" in out.stderr


def test_refuses_half_or_unreadable_tls(tmp_path):
    out = run({"VOID_API_TOKEN": "SYNTHETIC-token-1234567890", "VOID_TLS_CERT": str(tmp_path / "missing.pem")})
    assert out.returncode == 64 and "E_DEPLOY_TLS" in out.stderr


def test_dockerfile_runs_as_non_root_and_uses_the_pinned_requirements():
    text = (ENTRY.parent / "Dockerfile").read_text()
    assert "USER void" in text and "-r python/requirements.txt" in text and "ENTRYPOINT" in text
