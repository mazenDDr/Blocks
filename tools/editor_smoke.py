"""Run the real editor/native-state baseline with owned loopback services (ADR0026).

Evidence and the executed browser script stay outside the repository. No existing
workbench, backend, browser profile, Ollama process or credentials are used.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
EDITOR = ROOT / "apps/editor"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def chrome_path(explicit=None):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"Chrome is not executable: {path}")
        return str(path)
    for path in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                 shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chromium-browser")):
        if path and Path(path).is_file():
            return path
    raise ValueError("Install Chrome or supply --chrome /absolute/path/to/chrome; no browser is downloaded.")


def isolated_env():
    # Drop application knobs, provider credentials and injected plugins; retain OS tooling paths.
    return {k: v for k, v in os.environ.items()
            if not k.startswith(("VOID_", "ANTHROPIC_", "OPENAI_", "WANDB_", "MLFLOW_"))}


def stop(process):
    """Only groups created by this runner; never find/kill another user's port owner."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def wait_ready(url, process, headers=None, timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Service exited before readiness ({process.returncode}); see its log.")
        try:
            request = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(request, timeout=1) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(.1)
    raise RuntimeError(f"Service readiness exceeded {timeout}s: {url}")


def run(args, *, workbench=None, journey=None, extra_env=None, fixture="SYNTHETIC native model-free state workflow"):
    chrome = chrome_path(args.chrome)
    node, pnpm = shutil.which("node"), shutil.which("pnpm")
    if not node or not pnpm:
        raise ValueError("node and pnpm must be installed.")
    module = subprocess.check_output([node, "-p", "require.resolve('puppeteer-core')"], cwd=EDITOR, text=True).strip()
    if args.output:
        out = Path(args.output).expanduser().resolve()
        if out.is_relative_to(ROOT):
            raise ValueError("Evidence/output must be outside the repository.")
        out.mkdir(parents=True, exist_ok=False)
    else:
        out = Path(tempfile.mkdtemp(prefix="void-editor-smoke-"))
    env = isolated_env()
    token = secrets.token_urlsafe(32)  # ephemeral synthetic test credential, never written to evidence
    backend_port, editor_port = free_port(), free_port()
    while editor_port == backend_port:
        editor_port = free_port()
    base = f"http://127.0.0.1:{backend_port}"
    env.update(VOID_WORKBENCH=str(workbench or out / "workbench"), VOID_API_TOKEN=token, VOID_API=base)
    env.update(extra_env or {})
    processes, logs = [], []
    result = {"status": "failed", "fixture": fixture,
              "backend": base, "editor": f"http://127.0.0.1:{editor_port}", "chrome": chrome,
              "services": [], "cleanup": [], "error": None}
    started = time.monotonic()

    def start(name, command, cwd):
        log = (out / f"{name}.log").open("w")
        logs.append(log)
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append((name, proc))
        result["services"].append({"name": name, "pid": proc.pid})
        return proc

    previous = {}
    def interrupted(signum, frame):
        raise RuntimeError(f"Smoke interrupted by signal {signum}")
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, interrupted)
    try:
        backend = start("backend", [sys.executable, "-m", "uvicorn", "control.app:create_app", "--factory",
                          "--app-dir", "services", "--host", "127.0.0.1", "--port", str(backend_port)], ROOT)
        wait_ready(base+"/api/production", backend, {"Authorization": "Bearer "+token})
        editor = start("editor", [pnpm, "dev", "--host", "127.0.0.1", "--port", str(editor_port), "--strictPort"], EDITOR)
        wait_ready(result["editor"]+"/api/production", editor)
        script = out / "journey.mjs"
        shutil.copyfile(journey or EDITOR / "smoke/journey.mjs", script)
        env.update(VOID_SMOKE_URL=result["editor"], VOID_SMOKE_OUTPUT=str(out), VOID_SMOKE_CHROME=chrome,
                   VOID_SMOKE_PUPPETEER=module)
        browser = start("browser", [node, str(script)], out)
        code = browser.wait(timeout=args.timeout)
        if code:
            raise RuntimeError(f"Browser journey failed ({code}); see browser.log/evidence.json/failure.png.")
        evidence = json.loads((out / "evidence.json").read_text())
        if evidence.get("status") != "passed":
            raise RuntimeError("Browser did not record successful evidence.")
        result["status"] = "passed"
    except Exception as error:
        result["error"] = str(error)
    finally:
        for name, proc in reversed(processes):
            stop(proc)
            result["cleanup"].append({"name": name, "pid": proc.pid, "exitCode": proc.returncode})
        for log in logs:
            log.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        result["elapsedSeconds"] = time.monotonic()-started
        (out / "runner.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({**result, "evidenceDirectory": str(out)}, indent=2))
    return 0 if result["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="new evidence directory outside the repository; never overwrite")
    parser.add_argument("--chrome", default=os.environ.get("VOID_SMOKE_CHROME"))
    parser.add_argument("--timeout", type=float, default=180, help="browser journey seconds (1–600)")
    args = parser.parse_args()
    if not 1 <= args.timeout <= 600:
        parser.error("--timeout must be between 1 and 600 seconds")
    try:
        return run(args)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Smoke setup failed: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
