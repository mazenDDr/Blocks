"""Tools with bounded capabilities (VISION 12.1). A tool declares its effects; effects outside the process (`file_write`, `network`) are
EXTERNAL and require an explicit approval interrupt before they run, and they are executed through the effect ledger so a replay
after an interrupt or a restart cannot repeat them."""
from __future__ import annotations

import ast
import math
import operator
from pathlib import Path
from typing import Any, Callable

EXTERNAL_EFFECTS = ("file_write", "network")
MAX_READ_BYTES = 20000
MAX_WRITE_BYTES = 5000


class ToolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max}


def safe_eval(expr: str) -> float:
    """Arithmetic only: numbers, + - * / // % **, parentheses and a few named functions. No names, attributes or calls to anything else."""
    def ev(n: ast.AST) -> Any:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 100:
                raise ToolError("E_TOOL_ARGS", "exponent too large")
            return _BIN[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -ev(n.operand) if isinstance(n.op, ast.USub) else ev(n.operand)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ToolError("E_TOOL_ARGS", "only arithmetic is allowed in a calculator expression")
    try:
        return ev(ast.parse(expr.strip(), mode="eval"))
    except ToolError:
        raise
    except (SyntaxError, ZeroDivisionError, OverflowError, ValueError, TypeError) as e:
        raise ToolError("E_TOOL_ARGS", f"cannot evaluate expression: {type(e).__name__}")


def inside(base: Path, rel: str) -> Path:
    """Resolve `rel` under `base` and refuse anything that escapes it (including through symlinks and `..`)."""
    base = base.resolve()
    target = (base / rel).resolve()
    if target != base and base not in target.parents:
        raise ToolError("E_TOOL_PATH", f"path '{rel}' is outside the allowed directory")
    return target


class Tool:
    name: str
    description: str
    args: dict[str, str]
    effects: tuple[str, ...] = ()

    @property
    def external(self) -> bool:
        return any(e in EXTERNAL_EFFECTS for e in self.effects)

    def run(self, args: dict[str, Any], allowed_dir: Path) -> Any:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "args": self.args, "effects": list(self.effects), "external": self.external,
                "requiresApproval": self.external, "limits": self.limits()}

    def limits(self) -> dict[str, Any]:
        return {}


class Calculator(Tool):
    name, description, args, effects = "calculator", "Evaluate an arithmetic expression (no names, no calls).", {"expression": "text"}, ()

    def run(self, args, allowed_dir):
        v = safe_eval(str(args.get("expression", "")))
        return {"expression": args.get("expression"), "value": v}


class ReadTextFile(Tool):
    name, description, args, effects = "read_text_file", "Read a UTF-8 text file inside the allowed directory.", {"path": "text"}, ("file_read",)

    def limits(self):
        return {"maxBytes": MAX_READ_BYTES, "boundedTo": "the node's allowed directory"}

    def run(self, args, allowed_dir):
        p = inside(allowed_dir, str(args.get("path", "")))
        if not p.is_file():
            raise ToolError("E_TOOL_PATH", f"'{args.get('path')}' is not a file in the allowed directory")
        data = p.read_bytes()[:MAX_READ_BYTES]
        return {"path": str(args.get("path")), "text": data.decode("utf-8", "replace"), "truncated": p.stat().st_size > MAX_READ_BYTES}


class WriteNote(Tool):
    name, description, args, effects = "write_note", "Append a line of text to a file inside the allowed directory (an external effect).", {"path": "text", "text": "text"}, ("file_write",)

    def limits(self):
        return {"maxBytes": MAX_WRITE_BYTES, "boundedTo": "the node's allowed directory", "mode": "append"}

    def run(self, args, allowed_dir):
        text = str(args.get("text", ""))
        if len(text.encode()) > MAX_WRITE_BYTES:
            raise ToolError("E_TOOL_ARGS", f"text exceeds {MAX_WRITE_BYTES} bytes")
        allowed_dir.mkdir(parents=True, exist_ok=True)
        p = inside(allowed_dir, str(args.get("path", "")))
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(text + "\n")
        return {"path": str(args.get("path")), "bytesAppended": len(text.encode()) + 1}


TOOLS: dict[str, Tool] = {t.name: t for t in (Calculator(), ReadTextFile(), WriteNote())}
