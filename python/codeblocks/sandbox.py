"""Parent side of the code-block sandbox: one isolated child process per block instance (a long-lived session, so a training loop does not
pay interpreter start-up per call). The control service never executes user source; only the child does (see runner.py)."""
from __future__ import annotations

import atexit
import base64
import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import torch

RUNNER = Path(__file__).with_name("runner.py")
DTYPES = {"float32": torch.float32, "float64": torch.float64, "int64": torch.int64, "bool": torch.bool}
NAMES = {v: k for k, v in DTYPES.items()}
DEFAULT_LIMITS = {"cpu_seconds": 30, "wall_seconds": 30, "file_mb": 8}
_sessions: "set[SandboxSession]" = set()
_lock = threading.Lock()


class CodeBlockError(Exception):
    """A failure of the block's code or of its contract, with the stack mapped to the block's own source lines."""

    def __init__(self, code: str, message: str, frames: list[dict[str, Any]] | None = None, stdout: str = "", traceback_text: str = ""):
        super().__init__(message)
        self.code, self.message, self.frames, self.stdout, self.traceback = code, message, frames or [], stdout, traceback_text

    def to_json(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "frames": self.frames, "stdout": self.stdout, "traceback": self.traceback}


def enc(t: torch.Tensor) -> dict[str, Any]:
    t = t.detach().contiguous()
    return {"dtype": NAMES[t.dtype], "shape": list(t.shape), "data": base64.b64encode(t.numpy().tobytes()).decode()}


def dec(d: dict[str, Any]) -> torch.Tensor:
    raw = base64.b64decode(d["data"])
    dt = DTYPES[d["dtype"]]
    if not raw:
        return torch.empty(d["shape"], dtype=dt)
    return torch.frombuffer(bytearray(raw), dtype=dt).reshape(d["shape"]).clone()


_CODES = {"syntax": "E_CODE_SYNTAX", "exception": "E_CODE_EXCEPTION", "guard": "E_CODE_UNDECLARED_EFFECT", "output": "E_CODE_OUTPUT", "dependency": "E_CODE_DEPENDENCY",
          "entry": "E_CODE_ENTRY", "protocol": "E_CODE_PROTOCOL", "internal": "E_CODE_INTERNAL"}


class SandboxSession:
    def __init__(self, interface: dict[str, Any], source: str, label: str | None = None):
        self.interface, self.source = interface, source
        self.label = label or f"{interface['id']}@{interface['version']}"
        lim = dict(DEFAULT_LIMITS)
        lim.update({k: v for k, v in (interface.get("limits") or {}).items() if v})
        self.limits = lim
        self.proc: subprocess.Popen | None = None
        self.tmp: str | None = None
        self.stderr_path: str | None = None
        self.applied_limits: list[str] = []
        self.calls = 0

    # ------------------------------------------------------------------ process
    def start(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="void_block_")
        self.stderr_path = os.path.join(self.tmp, "stderr.txt")
        env = {"PATH": "/usr/bin:/bin", "HOME": self.tmp, "TMPDIR": self.tmp, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0",
               "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8"}  # nothing else of the parent's environment (tokens, credentials, ...) is inherited
        self._err = open(self.stderr_path, "wb")
        self.proc = subprocess.Popen([sys.executable, "-I", str(RUNNER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err, cwd=self.tmp, env=env,
                                     start_new_session=True, close_fds=True)
        with _lock:
            _sessions.add(self)
        r = self._rpc({"op": "load", "source": self.source, "label": self.label, "effects": self.interface.get("effects", []),
                       "randomness": self.interface.get("randomness", "none"), "differentiable": self.interface.get("differentiable", False),
                       "dependencies": self.interface.get("dependencies", []), "limits": self.limits})
        self.applied_limits = r.get("limits", [])

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def close(self) -> None:
        p, self.proc = self.proc, None
        if p is not None:
            try:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGKILL)
                p.wait(timeout=5)
            except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
                pass
            for f in (p.stdin, p.stdout):
                try:
                    f and f.close()
                except OSError:
                    pass
        try:
            self._err.close()
        except Exception:  # noqa: BLE001
            pass
        if self.tmp:
            shutil.rmtree(self.tmp, ignore_errors=True)
            self.tmp = None
        with _lock:
            _sessions.discard(self)

    # ------------------------------------------------------------------ rpc
    def _read_line(self, timeout: float) -> bytes | None:
        sel = selectors.DefaultSelector()
        sel.register(self.proc.stdout, selectors.EVENT_READ)
        buf = b""
        end = time.monotonic() + timeout
        fd = self.proc.stdout.fileno()
        try:
            while b"\n" not in buf:
                left = end - time.monotonic()
                if left <= 0 or not sel.select(left):
                    return None
                chunk = os.read(fd, 1 << 20)
                if not chunk:
                    return b""
                buf += chunk
        finally:
            sel.close()
        return buf

    def _died(self) -> CodeBlockError:
        rc = self.proc.poll() if self.proc else None
        if rc is None and self.proc is not None:
            try:
                rc = self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                rc = None
        why = f"exit code {rc}"
        if rc is not None and rc < 0:
            sig = -rc
            name = signal.Signals(sig).name if sig in set(int(s) for s in signal.Signals) else str(sig)
            why = {"SIGXCPU": "the CPU-time limit was exceeded", "SIGKILL": "the process was killed (memory limit or the system)", "SIGSEGV": "the process crashed (segmentation fault)",
                   "SIGXFSZ": "the file-size limit was exceeded"}.get(name, f"signal {name}")
        err = ""
        try:
            err = Path(self.stderr_path).read_text(errors="replace")[-2000:] if self.stderr_path else ""
        except OSError:
            pass
        code = "E_CODE_CPU_LIMIT" if "CPU-time" in why else "E_CODE_CRASH"
        self.close()
        return CodeBlockError(code, f"the block's process ended: {why}." + (f" stderr: {err.strip()}" if err.strip() else ""))

    def _rpc(self, req: dict[str, Any]) -> dict[str, Any]:
        if self.proc is None:
            raise CodeBlockError("E_CODE_CLOSED", "the sandbox session is closed")
        try:
            self.proc.stdin.write((json.dumps(req) + "\n").encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            raise self._died()
        timeout = float(self.limits.get("wall_seconds", 30)) if req["op"] != "load" else max(60.0, float(self.limits.get("wall_seconds", 30)))
        line = self._read_line(timeout)
        if line is None:
            self.close()
            raise CodeBlockError("E_CODE_TIMEOUT", f"the block did not answer within the wall-clock limit of {timeout:g} s; its process was killed")
        if line == b"":
            raise self._died()
        resp = json.loads(line)
        if not resp.get("ok"):
            e = resp["error"]
            raise CodeBlockError(_CODES.get(e["kind"], "E_CODE_EXCEPTION"), e["message"], e.get("frames", []), resp.get("stdout", ""), e.get("traceback", ""))
        return resp

    # ------------------------------------------------------------------ calls
    def ensure(self) -> None:
        if not self.alive():
            if self.proc is not None:
                self.close()
            self.start()

    def call(self, inputs: list[torch.Tensor], config: dict[str, Any], state: dict[str, torch.Tensor], seed: int | None, output_names: list[str],
             requires_grad: list[bool] | None = None) -> dict[str, Any]:
        self.ensure()
        self.calls += 1
        r = self._rpc({"op": "call", "inputs": [enc(t) for t in inputs], "config": config, "state": {k: enc(v) for k, v in state.items()}, "has_state": bool(self.interface.get("state")),
                       "seed": seed, "output_names": output_names, "requires_grad": requires_grad or [t.requires_grad for t in inputs]})
        return {"outputs": [dec(o) for o in r["outputs"]], "state": {k: dec(v) for k, v in r.get("state", {}).items()}, "stdout": r.get("stdout", ""),
                "rng_changed": r.get("rng_changed", False), "connected": r.get("connected_to_autograd", [])}

    def backward(self, inputs: list[torch.Tensor], config: dict[str, Any], state: dict[str, torch.Tensor], seed: int | None, output_names: list[str],
                 grad_outputs: list[torch.Tensor | None], requires_grad: list[bool]) -> list[torch.Tensor | None]:
        self.ensure()
        r = self._rpc({"op": "backward", "inputs": [enc(t) for t in inputs], "config": config, "state": {k: enc(v) for k, v in state.items()}, "has_state": bool(self.interface.get("state")),
                       "seed": seed, "output_names": output_names, "requires_grad": requires_grad,
                       "grad_outputs": [enc(g) if g is not None else None for g in grad_outputs]})
        return [dec(g) if g is not None else None for g in r["grads"]]

    def environment(self) -> dict[str, Any]:
        self.ensure()
        return self._rpc({"op": "env"})

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


@atexit.register
def _close_all() -> None:
    for s in list(_sessions):
        s.close()
