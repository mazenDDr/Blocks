"""Child process that executes the source of ONE code block. Started by codeblocks.sandbox as `python -I runner.py`.

This file deliberately imports nothing from the project: the child never sees the control service's modules, environment variables,
credentials or working directory (it runs in a scratch directory with a scrubbed environment).

Protocol: one JSON object per line on the real stdout; requests on stdin. Anything the user code prints is captured and returned, it can
never corrupt the protocol. Resource limits (CPU seconds, address space, file size, open files) are applied at start-up; the parent also
enforces a wall-clock timeout and kills the process group.

Effect guards (network, file write, file read, subprocesses) are monkey-patches installed only while user code runs. They turn accidental
undeclared effects into clear errors; they are NOT a security boundary against hostile code. The boundary is the separate OS process with
limits and a scrubbed environment."""
from __future__ import annotations

import base64
import builtins
import contextlib
import hashlib
import io
import json
import os
import sys
import traceback

_REAL_STDOUT = os.fdopen(os.dup(1), "w", buffering=1)
sys.stdout = io.StringIO()  # user prints go here
os.dup2(os.open(os.devnull, os.O_WRONLY), 1)

import torch  # noqa: E402

DTYPES = {"float32": torch.float32, "float64": torch.float64, "int64": torch.int64, "bool": torch.bool}
NAMES = {v: k for k, v in DTYPES.items()}
FILENAME = "<block>"
NS: dict = {}
LOADED: dict = {}


def enc(t: torch.Tensor) -> dict:
    t = t.detach().contiguous()
    return {"dtype": NAMES[t.dtype], "shape": list(t.shape), "data": base64.b64encode(t.numpy().tobytes()).decode()}


def dec(d: dict) -> torch.Tensor:
    shape, dt = d["shape"], DTYPES[d["dtype"]]
    raw = base64.b64decode(d["data"])
    if not raw:
        return torch.empty(shape, dtype=dt)
    return torch.frombuffer(bytearray(raw), dtype=dt).reshape(shape).clone()


class GuardError(PermissionError):
    pass


@contextlib.contextmanager
def guards(effects: set):
    import os as _os
    import socket
    import subprocess

    saved = []

    def patch(obj, name, repl):
        if hasattr(obj, name):
            saved.append((obj, name, getattr(obj, name)))
            setattr(obj, name, repl)

    def deny(what):
        def f(*a, **k):
            raise GuardError(f"{what} is not allowed: the block did not declare this effect (declared effects: {sorted(effects) or 'none'})")
        return f

    for n in ("system", "fork", "forkpty", "execv", "execve", "execvp", "execl", "execlp", "posix_spawn", "spawnv", "spawnl", "popen", "kill"):
        patch(_os, n, deny(f"os.{n}"))
    patch(subprocess, "Popen", deny("starting a subprocess"))
    if "network" not in effects:
        for n in ("connect", "connect_ex", "bind", "sendto"):
            patch(socket.socket, n, deny(f"network access (socket.{n})"))
        patch(socket, "getaddrinfo", deny("network access (DNS lookup)"))
    real_open = builtins.open

    def guarded_open(file, mode="r", *a, **k):
        writing = any(c in str(mode) for c in "wax+")
        if writing and "file_write" not in effects:
            raise GuardError(f"writing a file ('{file}') is not allowed: the block did not declare file_write")
        if not writing and "file_read" not in effects:
            raise GuardError(f"reading a file ('{file}') is not allowed: the block did not declare file_read")
        return real_open(file, mode, *a, **k)

    patch(builtins, "open", guarded_open)
    if "file_write" not in effects:
        for n in ("remove", "unlink", "rename", "replace", "mkdir", "makedirs", "rmdir", "truncate"):
            patch(_os, n, deny(f"os.{n} (file write)"))
    try:
        yield
    finally:
        for obj, name, orig in reversed(saved):
            setattr(obj, name, orig)


def frames_of(exc: BaseException) -> list:
    out = []
    for fs in traceback.extract_tb(exc.__traceback__):
        if fs.filename == FILENAME:
            line = LOADED.get("lines", [])
            out.append({"line": fs.lineno, "function": fs.name, "text": line[fs.lineno - 1].rstrip() if 0 < (fs.lineno or 0) <= len(line) else ""})
    if isinstance(exc, SyntaxError) and exc.filename == FILENAME:
        out.append({"line": exc.lineno, "function": "<module>", "text": (exc.text or "").rstrip(), "offset": exc.offset})
    return out


def error(kind: str, exc: BaseException | None = None, message: str | None = None) -> dict:
    e = {"kind": kind, "message": message if message is not None else f"{type(exc).__name__}: {exc}", "type": type(exc).__name__ if exc else kind,
         "frames": frames_of(exc) if exc else []}
    if exc is not None:
        e["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).replace(
            f'File "{FILENAME}"', f'File "{LOADED.get("label", "block")}"')
    return {"ok": False, "error": e, "stdout": sys.stdout.getvalue()[-20000:]}


def apply_limits(limits: dict) -> list:
    applied = []
    try:
        import resource
    except ImportError:
        return ["resource module unavailable"]
    plan = [("RLIMIT_CPU", int(limits.get("cpu_seconds", 30)), "cpu_seconds"),
            ("RLIMIT_FSIZE", int(limits.get("file_mb", 8)) * 1024 * 1024, "file_mb"),
            ("RLIMIT_NOFILE", 256, "open_files"),
            ("RLIMIT_CORE", 0, "core")]
    mem = int(limits.get("memory_mb", 0))
    if mem:
        plan.append(("RLIMIT_AS", mem * 1024 * 1024, "memory_mb"))
    for name, val, label in plan:
        try:
            lim = getattr(resource, name)
            soft, hard = resource.getrlimit(lim)
            new = val if hard in (-1, resource.RLIM_INFINITY) else min(val, hard)
            resource.setrlimit(lim, (new, hard))
            applied.append(f"{label}={val}")
        except (ValueError, OSError, AttributeError) as e:
            applied.append(f"{label}: not enforced on this platform ({e})")
    return applied


def seed_all(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2 ** 32))
    except Exception:  # noqa: BLE001
        pass


def rng_fingerprint() -> str:
    import random
    h = hashlib.sha256(bytes(torch.get_rng_state().tolist()))
    h.update(repr(random.getstate()).encode())
    try:
        import numpy as np
        h.update(repr(np.random.get_state()[1][:5].tolist()).encode())
    except Exception:  # noqa: BLE001
        pass
    return h.hexdigest()


def check_deps(deps: list) -> list:
    from importlib import metadata
    bad = []
    for d in deps:
        name, _, want = d.partition("==")
        try:
            have = metadata.version(name.strip())
        except metadata.PackageNotFoundError:
            bad.append(f"{d}: not installed")
            continue
        if want and have != want.strip():
            bad.append(f"{d}: installed version is {have}")
    return bad


def do_load(req: dict) -> dict:
    LOADED.clear()
    NS.clear()
    LOADED.update(effects=set(req.get("effects", [])), lines=req["source"].splitlines(), label=req.get("label", "block"),
                  randomness=req.get("randomness", "none"), differentiable=bool(req.get("differentiable")))
    applied = apply_limits(req.get("limits", {}))
    bad = check_deps(req.get("dependencies", []))
    if bad:
        return {"ok": False, "error": {"kind": "dependency", "message": "pinned dependencies are not satisfied: " + "; ".join(bad), "type": "DependencyError", "frames": []}}
    try:
        code = compile(req["source"], FILENAME, "exec")
    except SyntaxError as e:
        return error("syntax", e)
    NS.update({"__name__": "void_block", "__builtins__": builtins})
    try:
        with guards(LOADED["effects"]):
            exec(code, NS)  # noqa: S102  (this IS the sandboxed child: executing the user's block is its job)
    except BaseException as e:  # noqa: BLE001
        return error("exception", e)
    if not callable(NS.get("run")):
        return {"ok": False, "error": {"kind": "entry", "message": "the source must define a function `run`", "type": "EntryPointError", "frames": []}}
    return {"ok": True, "limits": applied}


def run_block(req: dict, backward: bool) -> dict:
    inputs = [dec(d) for d in req["inputs"]]
    state = {k: dec(v) for k, v in req.get("state", {}).items()}
    config = req.get("config", {})
    diff = LOADED["differentiable"]
    if req.get("seed") is not None:
        seed_all(int(req["seed"]))
    before = rng_fingerprint()
    float_idx = [i for i, t in enumerate(inputs) if t.dtype.is_floating_point]
    if diff:
        for i in float_idx:
            inputs[i].requires_grad_(bool(req.get("requires_grad", [True] * len(inputs))[i]))
    kwargs = dict(config)
    if req.get("has_state"):
        kwargs["state"] = state
    try:
        with guards(LOADED["effects"]), torch.set_grad_enabled(diff):
            result = NS["run"](*inputs, **kwargs)
    except BaseException as e:  # noqa: BLE001
        return error("guard" if isinstance(e, GuardError) else "exception", e)
    after = rng_fingerprint()
    names = req["output_names"]
    if isinstance(result, torch.Tensor):
        result = [result]
    if isinstance(result, dict):
        if set(result) != set(names):
            return error("output", message=f"run returned outputs {sorted(result)} but the interface declares {sorted(names)}")
        outs = [result[n] for n in names]
    elif isinstance(result, (tuple, list)) and len(result) == len(names):
        outs = list(result)
    else:
        return error("output", message=f"run must return a dict {names}, a tensor, or a tuple of {len(names)} tensors; it returned {type(result).__name__}")
    for n, o in zip(names, outs):
        if not isinstance(o, torch.Tensor):
            return error("output", message=f"output '{n}' is a {type(o).__name__}, not a torch.Tensor")
    resp = {"ok": True, "stdout": sys.stdout.getvalue()[-20000:], "rng_changed": before != after}
    if backward:
        grad_out = [dec(g) if g is not None else None for g in req["grad_outputs"]]
        diffable = [(o, g) for o, g in zip(outs, grad_out) if o.requires_grad and g is not None]
        wrt = [inputs[i] for i in float_idx if inputs[i].requires_grad]
        if not diffable or not wrt:
            resp["grads"] = [None] * len(inputs)
            resp["warning"] = "no output is connected to a differentiable input through torch operations"
            return resp
        try:
            gs = torch.autograd.grad([o for o, _ in diffable], wrt, [g for _, g in diffable], allow_unused=True)
        except BaseException as e:  # noqa: BLE001
            return error("exception", e)
        it = iter(gs)
        grads = [None] * len(inputs)
        for i in float_idx:
            if inputs[i].requires_grad:
                g = next(it)
                grads[i] = enc(g) if g is not None else None
        resp["grads"] = grads
        return resp
    resp["outputs"] = [enc(o) for o in outs]
    resp["connected_to_autograd"] = [bool(o.requires_grad) for o in outs]
    resp["state"] = {k: enc(v) for k, v in state.items() if isinstance(v, torch.Tensor)}
    return resp


def main() -> None:
    for line in sys.stdin:
        sys.stdout = io.StringIO()
        try:
            req = json.loads(line)
            op = req["op"]
            if op == "load":
                resp = do_load(req)
            elif op == "call":
                resp = run_block(req, backward=False)
            elif op == "backward":
                resp = run_block(req, backward=True)
            elif op == "env":
                from importlib import metadata
                resp = {"ok": True, "python": sys.version.split()[0], "torch": torch.__version__,
                        "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions() if d.metadata["Name"])}
            elif op == "exit":
                return
            else:
                resp = {"ok": False, "error": {"kind": "protocol", "message": f"unknown op {op}", "type": "ProtocolError", "frames": []}}
        except BaseException as e:  # noqa: BLE001
            resp = error("internal", e)
        _REAL_STDOUT.write(json.dumps(resp) + "\n")
        _REAL_STDOUT.flush()


if __name__ == "__main__":
    main()
