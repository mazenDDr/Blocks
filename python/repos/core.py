"""Pinned Git repository browsing and code import (VISION §7.9, acceptance A44).

A repository is fetched into a bare mirror under the workbench (`repos/<sha256(url)>.git`). Nothing is ever checked out, installed or
imported in this process: trees come from `git ls-tree`, file bytes from `git cat-file blob` (no smudge/textconv filters), comparisons from
`git diff --no-ext-diff --no-textconv`, Python files are read with `ast.parse` only, and dependency declarations are parsed as text/TOML/INI.
`setup.py` and other build scripts are reported as executable installation code that was not run. Git runs with hooks disabled, a
fixed protocol allowlist (file, https, ssh; never `ext::`), no terminal prompts and no system config. URLs that embed credentials are
refused; authorized remotes use the environment's own Git credential configuration.

Importing pins one Python file at an exact commit and wraps one of its top-level functions as an ordinary code block. Execution of the
imported code happens only when that block is tested or run, in the existing isolated code-block sandbox (ADR 0007). The import record
(URL, commit, tree, path, blob, sha256, license file, declared dependencies, selected pins, entry point, generated source hash) is
content-addressed and immutable; the block carries it as `origin`, so the pin is part of the graph's semantic hash."""
from __future__ import annotations

import ast
import configparser
import hashlib
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

MAX_FILE_BYTES = 256 * 1024
MAX_TREE_ENTRIES = 20_000
GIT_TIMEOUT = 120
_SHA = re.compile(r"^[0-9a-f]{40}$")
_REV = re.compile(r"^[A-Za-z0-9._/@^~-]{1,200}$")
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MAX_BUNDLE_MODULES = 50
MAX_BUNDLE_BYTES = 1_000_000
INSTALL_SCRIPTS = {"setup.py", "conftest.py", "noxfile.py", "tasks.py", "build.py", "Makefile", "install.sh", "setup.sh"}


class RepoError(Exception):
    def __init__(self, code: str, message: str, status: int = 422):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def check_url(url: str) -> str:
    """Accept a local path, file://, https:// or ssh remote. Refuse embedded credentials and transport helpers."""
    url = url.strip()
    if not url or len(url) > 1000 or url.startswith("-") or any(c in url for c in "\n\r\0"):
        raise RepoError("E_REPO_URL", "Give a repository path or a file://, https:// or ssh:// URL.")
    if "::" in url.split("/")[0]:
        raise RepoError("E_REPO_URL", "Git transport helpers (such as ext::) are not allowed.")
    if "://" in url:
        parts = urlsplit(url)
        if parts.scheme not in ("file", "https", "ssh"):
            raise RepoError("E_REPO_URL", f"Scheme '{parts.scheme}' is not allowed; use file, https or ssh.")
        if parts.password is not None or (parts.scheme == "https" and parts.username):
            raise RepoError("E_REPO_CREDENTIALS", "The URL embeds credentials. Remove them and configure Git credentials in the environment instead; "
                            "credentials are never stored in a project or import record.")
    elif re.match(r"^[\w.-]+@[\w.-]+:", url):
        pass  # scp-like ssh syntax (git@host:org/repo.git): authentication comes from the ssh agent, nothing is embedded
    elif not Path(url).is_absolute():
        raise RepoError("E_REPO_URL", "A local repository must be an absolute path.")
    return url


class Repos:
    def __init__(self, root: str | Path):
        self.root = Path(root) / "repos"
        self.root.mkdir(parents=True, exist_ok=True)
        self.imports = self.root / "imports"
        self.imports.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ git plumbing
    def _git(self, mirror: Path | None, *args: str, input_: bytes | None = None, check: bool = True) -> bytes:
        # HOME is kept so the user's own Git credential helpers/ssh settings authorize private remotes; the safety settings below are
        # passed with -c, which takes precedence over any user configuration.
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", str(self.root)), "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1",
               "GIT_ASKPASS": "", "SSH_ASKPASS": "", "LC_ALL": "C"}
        for k in ("SSH_AUTH_SOCK", "GIT_SSH_COMMAND", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy", "SSL_CERT_FILE", "GIT_SSL_CAINFO"):
            if k in os.environ:
                env[k] = os.environ[k]
        hard = ["-c", "core.hooksPath=/dev/null", "-c", "protocol.allow=never", "-c", "protocol.file.allow=always", "-c", "protocol.https.allow=always",
                "-c", "protocol.ssh.allow=always", "-c", "core.fsmonitor=false", "-c", "submodule.recurse=false", "-c", "core.symlinks=false"]
        cmd = ["git", *hard, *(["--git-dir", str(mirror)] if mirror else []), *args]
        try:
            p = subprocess.run(cmd, input=input_, capture_output=True, env=env, timeout=GIT_TIMEOUT, cwd=self.root)
        except subprocess.TimeoutExpired as e:
            raise RepoError("E_REPO_TIMEOUT", f"git {args[0]} did not finish within {GIT_TIMEOUT} s.", 504) from e
        if check and p.returncode != 0:
            raise RepoError("E_REPO_GIT", f"git {args[0]} failed: {p.stderr.decode(errors='replace').strip()[-600:]}")
        return p.stdout

    def mirror_of(self, url: str) -> Path:
        return self.root / f"{_sha256(url.encode())[:24]}.git"

    def repo_id(self, url: str) -> str:
        return _sha256(url.encode())[:24]

    def _mirror_by_id(self, repo_id: str) -> tuple[Path, str]:
        if not re.fullmatch(r"[0-9a-f]{24}", repo_id):
            raise RepoError("E_REPO_UNKNOWN", "Unknown repository id.", 404)
        m = self.root / f"{repo_id}.git"
        meta = m / "void-source.json"
        if not meta.exists():
            raise RepoError("E_REPO_UNKNOWN", "This repository has not been fetched in this workbench.", 404)
        return m, json.loads(meta.read_text())["url"]

    # ------------------------------------------------------------------ resolve
    def resolve(self, url: str, rev: str = "HEAD") -> dict[str, Any]:
        """Fetch branches/tags (never checking out) and resolve `rev` to a commit."""
        url = check_url(url)
        if not _REV.match(rev) or rev.startswith("-"):
            raise RepoError("E_REPO_REV", "Revision must be a branch, tag, HEAD or a commit SHA.")
        m = self.mirror_of(url)
        if not m.exists():
            self._git(None, "init", "--bare", "--quiet", str(m))
            (m / "void-source.json").write_text(json.dumps({"url": url}))
        self._git(m, "fetch", "--quiet", "--no-tags", "--prune", "--no-recurse-submodules", "--", url,
                  "+HEAD:refs/void/HEAD", "+refs/heads/*:refs/void/heads/*", "+refs/tags/*:refs/void/tags/*")
        refs = {}
        for line in self._git(m, "for-each-ref", "--format=%(objectname) %(refname)", "refs/void/").decode().splitlines():
            sha, ref = line.split(" ", 1)
            refs[ref.removeprefix("refs/void/")] = sha
        commit = None
        if _SHA.match(rev):
            ok = self._git(m, "cat-file", "-t", rev, check=False).decode().strip() == "commit"
            commit = rev if ok else None
        else:
            for cand in (rev, f"heads/{rev}", f"tags/{rev}"):
                if cand in refs:
                    commit = self._git(m, "rev-parse", "--verify", f"{refs[cand]}^{{commit}}").decode().strip()
                    break
        if commit is None:
            raise RepoError("E_REPO_REV", f"Revision '{rev}' was not found in {url}.")
        info = self._git(m, "show", "-s", "--format=%T%n%an%n%cI%n%s", commit).decode().split("\n")
        return {"repoId": self.repo_id(url), "url": url, "requested": rev, "commit": commit, "tree": info[0], "author": info[1], "committedAt": info[2],
                "subject": info[3], "refs": {k: v for k, v in sorted(refs.items())},
                "note": "Fetched into a bare mirror; nothing was checked out, installed or executed."}

    def _commit(self, m: Path, commit: str) -> str:
        if not _SHA.match(commit) or self._git(m, "cat-file", "-t", commit, check=False).decode().strip() != "commit":
            raise RepoError("E_REPO_REV", "Give a full 40-character commit SHA from resolve().")
        return commit

    # ------------------------------------------------------------------ browse
    def tree(self, repo_id: str, commit: str) -> dict[str, Any]:
        m, url = self._mirror_by_id(repo_id)
        commit = self._commit(m, commit)
        raw = self._git(m, "ls-tree", "-r", "-z", "--full-tree", "-l", commit)
        entries = []
        for rec in raw.split(b"\0"):
            if not rec:
                continue
            head, path = rec.split(b"\t", 1)
            mode, typ, obj, size = head.decode().split()
            p = path.decode("utf-8", "replace")
            entries.append({"path": p, "mode": mode, "type": typ, "blob": obj, "size": None if size == "-" else int(size), "kind": self._classify(m, p, mode, typ, obj, size)})
            if len(entries) > MAX_TREE_ENTRIES:
                raise RepoError("E_REPO_TOO_LARGE", f"More than {MAX_TREE_ENTRIES} entries; browsing this repository is not supported.")
        counts: dict[str, int] = {}
        for e in entries:
            counts[e["kind"]] = counts.get(e["kind"], 0) + 1
        return {"repoId": repo_id, "url": url, "commit": commit, "entries": entries, "counts": counts,
                "dependencies": self.dependencies(repo_id, commit, entries), "license": self.license(repo_id, commit, entries)}

    def _classify(self, m: Path, path: str, mode: str, typ: str, obj: str, size: str) -> str:
        name = path.rsplit("/", 1)[-1]
        if typ == "commit":
            return "submodule"  # gitlink: recorded, never fetched or recursed into
        if mode == "120000":
            return "symlink"  # never followed
        if name in INSTALL_SCRIPTS or (name.endswith(".sh") and "/" not in path):
            return "installation_script"
        if size != "-" and int(size) < 400:
            head = self._git(m, "cat-file", "blob", obj)
            if head.startswith(b"version https://git-lfs.github.com/spec/"):
                return "lfs_pointer"
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if name in ("requirements.txt", "pyproject.toml", "setup.cfg", "environment.yml", "Pipfile", "poetry.lock", "uv.lock") or name.startswith("requirements"):
            return "dependency_manifest"
        if name.upper().startswith(("LICENSE", "LICENCE", "COPYING")):
            return "license"
        if ext == "py":
            return "python_source"
        if ext == "ipynb":
            return "notebook"
        if ext in ("csv", "tsv", "parquet", "jsonl", "npy", "npz", "h5", "hdf5", "arrow", "feather"):
            return "dataset"
        if ext in ("pt", "pth", "ckpt", "safetensors", "onnx", "pkl", "joblib", "keras", "bin"):
            return "model_weights"
        if ext in ("yaml", "yml", "json", "toml", "ini", "cfg"):
            return "configuration"
        if ext in ("md", "rst", "txt"):
            return "documentation"
        return "other"

    def _entry(self, m: Path, commit: str, path: str) -> dict[str, str]:
        if path.startswith("/") or ".." in path.split("/") or "\0" in path or not path:
            raise RepoError("E_REPO_PATH", "Give a repository-relative path.")
        raw = self._git(m, "ls-tree", "-z", "--full-tree", commit, "--", path)
        if not raw:
            raise RepoError("E_REPO_PATH", f"'{path}' does not exist at commit {commit[:12]}.", 404)
        head, _ = raw.rstrip(b"\0").split(b"\t", 1)
        mode, typ, obj = head.decode().split()
        if typ != "blob" or mode == "120000":
            raise RepoError("E_REPO_PATH", f"'{path}' is a {'symlink' if mode == '120000' else typ}, not a regular file.")
        return {"mode": mode, "blob": obj}

    def read(self, repo_id: str, commit: str, path: str) -> dict[str, Any]:
        m, url = self._mirror_by_id(repo_id)
        commit = self._commit(m, commit)
        e = self._entry(m, commit, path)
        data = self._git(m, "cat-file", "blob", e["blob"])  # raw blob: no smudge/clean filters, no LFS download
        out = {"repoId": repo_id, "url": url, "commit": commit, "path": path, "blob": e["blob"], "sha256": _sha256(data), "bytes": len(data)}
        if len(data) > MAX_FILE_BYTES:
            return out | {"text": None, "note": f"Larger than {MAX_FILE_BYTES} bytes; content not shown."}
        try:
            return out | {"text": data.decode("utf-8")}
        except UnicodeDecodeError:
            return out | {"text": None, "note": "Binary content; not shown."}

    # ------------------------------------------------------------------ static metadata (never executed)
    def dependencies(self, repo_id: str, commit: str, entries: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        m, _ = self._mirror_by_id(repo_id)
        entries = entries if entries is not None else self.tree(repo_id, commit)["entries"]
        found, notes = [], []
        for e in entries:
            name, path = e["path"].rsplit("/", 1)[-1], e["path"]
            if e["type"] != "blob" or (e["size"] or 0) > MAX_FILE_BYTES:
                continue
            if name.startswith("requirements") and name.endswith(".txt"):
                text = self._git(m, "cat-file", "blob", e["blob"]).decode("utf-8", "replace")
                for line in text.splitlines():
                    line = line.split("#", 1)[0].strip()
                    if not line:
                        continue
                    if line.startswith("-"):
                        notes.append(f"{path}: option line '{line[:60]}' not followed (no installation is performed)")
                        continue
                    found.append(_req(line, path))
            elif name == "pyproject.toml":
                try:
                    data = tomllib.loads(self._git(m, "cat-file", "blob", e["blob"]).decode("utf-8", "replace"))
                except tomllib.TOMLDecodeError as ex:
                    notes.append(f"{path}: invalid TOML ({ex})")
                    continue
                proj = data.get("project", {})
                found += [_req(r, path) for r in proj.get("dependencies", []) if isinstance(r, str)]
                for extra, reqs in (proj.get("optional-dependencies") or {}).items():
                    found += [_req(r, f"{path} [{extra}]") for r in reqs if isinstance(r, str)]
                if "build-system" in data:
                    notes.append(f"{path}: declares a build backend ({data['build-system'].get('build-backend', 'unspecified')}); it was not run")
            elif name == "setup.cfg":
                cp = configparser.ConfigParser(interpolation=None)
                try:
                    cp.read_string(self._git(m, "cat-file", "blob", e["blob"]).decode("utf-8", "replace"))
                except configparser.Error as ex:
                    notes.append(f"{path}: unreadable ({ex})")
                    continue
                if cp.has_option("options", "install_requires"):
                    found += [_req(r.strip(), path) for r in cp.get("options", "install_requires").splitlines() if r.strip()]
            elif name == "setup.py":
                notes.append(f"{path}: executable installation script; NOT run, so any dependencies it computes are not extracted")
            elif name in ("environment.yml", "Pipfile", "poetry.lock", "uv.lock"):
                notes.append(f"{path}: present; this format is not parsed")
        return {"declared": found, "notes": notes}

    def license(self, repo_id: str, commit: str, entries: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        m, _ = self._mirror_by_id(repo_id)
        entries = entries if entries is not None else self.tree(repo_id, commit)["entries"]
        lic = [e for e in entries if "/" not in e["path"] and e["kind"] == "license"]
        if not lic:
            return None
        e = sorted(lic, key=lambda x: x["path"])[0]
        text = self._git(m, "cat-file", "blob", e["blob"])
        t = " ".join(text.decode("utf-8", "replace").split()).lower()
        guess = ("MIT" if "permission is hereby granted, free of charge" in t else
                 "Apache-2.0" if "apache license" in t and "version 2.0" in t else
                 "GPL-3.0" if "gnu general public license" in t and "version 3" in t else
                 "BSD-3-Clause" if "redistribution and use" in t and "neither the name" in t else
                 "BSD-2-Clause" if "redistribution and use" in t else None)
        return {"path": e["path"], "blob": e["blob"], "sha256": _sha256(text), "spdxGuess": guess,
                "note": "Identified from the license text by keyword; verify the actual terms before reuse."}

    def inspect_python(self, repo_id: str, commit: str, path: str) -> dict[str, Any]:
        """Parse (never import) a Python file: top-level functions with parameters, imports, and what blocks wrapping it."""
        f = self.read(repo_id, commit, path)
        if not path.endswith(".py") or f["text"] is None:
            raise RepoError("E_REPO_NOT_PYTHON", f"'{path}' is not a readable Python source file.")
        try:
            tree = ast.parse(f["text"], filename=path)
        except SyntaxError as e:
            raise RepoError("E_REPO_SYNTAX", f"{path} line {e.lineno}: {e.msg}") from e
        entries = self.tree(repo_id, commit)["entries"]
        index = _module_index(entries)
        imports, local = [], []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                imports.append(node.module or "")
        modules, unresolved = self._closure(repo_id, commit, path, tree, index)
        local = sorted({m for m in modules if m != _module_name(path)})
        funcs = []
        for n in tree.body:
            if isinstance(n, ast.FunctionDef):
                a = n.args
                defaults = [None] * (len(a.args) - len(a.defaults)) + [ast.literal_eval(d) if _literal(d) else "<expression>" for d in a.defaults]
                funcs.append({"name": n.name, "line": n.lineno, "params": [{"name": x.arg, "default": d, "hasDefault": i >= len(a.args) - len(a.defaults)}
                                                                         for i, (x, d) in enumerate(zip(a.args, defaults))],
                              "varargs": bool(a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs), "doc": ast.get_docstring(n)})
        problems = []
        if unresolved:
            problems.append(f"imports repository modules that cannot be resolved at this commit ({', '.join(sorted(unresolved))})")
        bundle = [modules[m] for m in local]
        if len(bundle) > MAX_BUNDLE_MODULES or sum(b["bytes"] for b in bundle) > MAX_BUNDLE_BYTES:
            problems.append(f"its repository modules exceed the bundle limit ({MAX_BUNDLE_MODULES} files, {MAX_BUNDLE_BYTES} bytes)")
        if any(f_["name"] == "run" for f_ in funcs):
            problems.append("defines a top-level 'run', which is the code-block entry name; wrapping is refused")
        return {**{k: f[k] for k in ("repoId", "url", "commit", "path", "blob", "sha256")}, "functions": funcs, "imports": sorted(set(imports)),
                "localImports": local, "bundledModules": bundle, "wrappable": not problems, "problems": problems,
                "note": "Parsed with ast; the module was not imported or executed."}

    def _closure(self, repo_id, commit, path, tree, index):
        """Repository modules reachable from `path` by (relative or absolute) imports, parsed with ast only. Package __init__ files of every
        imported module are included because Python executes them on import. Returns ({module: entry}, unresolved relative imports)."""
        found, unresolved, queue = {}, set(), [(_module_name(path), path, tree)]
        seen = {_module_name(path)}
        while queue:
            mod, p, t = queue.pop()
            pkg = mod if p.endswith("/__init__.py") else mod.rpartition(".")[0]
            wanted = []
            for node in ast.walk(t):
                if isinstance(node, ast.Import):
                    wanted += [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:  # relative: resolve against this module's package, `level - 1` packages up
                        base, up = (pkg.split(".") if pkg else []), node.level - 1
                        target = ".".join([*base[:len(base) - up], *(node.module.split(".") if node.module else [])]) if up <= len(base) else ""
                        if not base or not target:
                            unresolved.add("." * node.level + (node.module or ""))
                            continue
                    else:
                        target = node.module or ""
                    wanted.append(target)
                    wanted += [f"{target}.{a.name}" for a in node.names if a.name != "*"]  # `from pkg import submodule`
            for name in wanted:
                parts = name.split(".")
                for i in range(1, len(parts) + 1):  # parent packages run first on import
                    m = ".".join(parts[:i])
                    if m in index and m not in seen:
                        seen.add(m)
                        info = self.read(repo_id, commit, index[m])
                        if info["text"] is None:
                            unresolved.add(m)
                            continue
                        found[m] = {"module": m, "path": index[m], "blob": info["blob"], "sha256": info["sha256"], "bytes": info["bytes"],
                                    "package": index[m].endswith("/__init__.py"), "text": info["text"]}
                        try:
                            queue.append((m, index[m], ast.parse(info["text"], filename=index[m])))
                        except SyntaxError:
                            unresolved.add(m)
        return found, unresolved

    # ------------------------------------------------------------------ compare / update deliberately
    def compare(self, repo_id: str, base: str, head: str, paths: list[str]) -> dict[str, Any]:
        m, url = self._mirror_by_id(repo_id)
        base, head = self._commit(m, base), self._commit(m, head)
        for p in paths:
            if p.startswith("/") or ".." in p.split("/") or p.startswith("-"):
                raise RepoError("E_REPO_PATH", "Give repository-relative paths.")
        status = self._git(m, "diff", "--no-ext-diff", "--no-textconv", "--name-status", "-z", base, head, "--", *paths).split(b"\0")
        changed = [{"status": status[i].decode(), "path": status[i + 1].decode("utf-8", "replace")} for i in range(0, len(status) - 1, 2) if status[i]]
        patch = self._git(m, "diff", "--no-ext-diff", "--no-textconv", "--no-color", base, head, "--", *paths)[:MAX_FILE_BYTES].decode("utf-8", "replace")
        return {"repoId": repo_id, "url": url, "base": base, "head": head, "paths": paths, "changed": changed, "patch": patch}

    # ------------------------------------------------------------------ import as a code block
    def import_function(self, repo_id: str, commit: str, path: str, function: str, interface: dict[str, Any], pins: list[str] | None = None) -> dict[str, Any]:
        info = self.inspect_python(repo_id, commit, path)
        if not info["wrappable"]:
            raise RepoError("E_REPO_NOT_WRAPPABLE", f"{path} cannot be wrapped: " + "; ".join(info["problems"]))
        fn = next((f for f in info["functions"] if f["name"] == function), None)
        if fn is None:
            raise RepoError("E_REPO_ENTRY", f"{path} has no top-level function '{function}'.")
        if fn["varargs"]:
            raise RepoError("E_REPO_ENTRY", f"'{function}' uses *args/**kwargs/keyword-only/positional-only parameters; declare a plain function.")
        inputs = [i["name"] for i in interface.get("inputs", [])]
        config = [c["name"] for c in interface.get("config", [])]
        outputs = [o["name"] for o in interface.get("outputs", [])]
        if not inputs or not outputs or not all(_NAME.match(x) for x in inputs + config + outputs):
            raise RepoError("E_REPO_INTERFACE", "Declare at least one input and one output with Python identifier names.")
        params = [p["name"] for p in fn["params"]]
        unknown = [x for x in inputs + config if x not in params]
        required = [p["name"] for p in fn["params"] if not p["hasDefault"] and p["name"] not in inputs + config]
        if unknown or required:
            raise RepoError("E_REPO_INTERFACE", f"Interface does not match {function}({', '.join(params)}): "
                            + "; ".join(([f"not parameters: {unknown}"] if unknown else []) + ([f"required parameters not supplied: {required}"] if required else [])))
        text = self.read(repo_id, commit, path)["text"]
        bundle = info["bundledModules"]
        preamble = _bundle_preamble(bundle, _module_name(path), path.endswith("/__init__.py")) if bundle or "/" in path else ""
        call = f"{function}({', '.join(f'{x}={x}' for x in inputs + config)})"
        sig = ", ".join(inputs) + (", *, " + ", ".join(f"{c['name']}={c.get('default')!r}" for c in interface.get("config", [])) if config else "")
        ret = f"{{{outputs[0]!r}: _out}}" if len(outputs) == 1 else "{" + ", ".join(f"{o!r}: _out[{i}]" for i, o in enumerate(outputs)) + "}"
        source = (f"# Imported from {info['url']} at commit {commit}, file {path} (blob {info['blob']}, sha256 {info['sha256']}).\n"
                  "# Edits to this block are recorded as local modifications of the pinned import.\n"
                  + preamble + text.rstrip("\n") + "\n\n\n# --- workbench adapter (generated) ---\n"
                  f"def run({sig}):\n    _out = {call}\n" + (f"    if not isinstance(_out, tuple) or len(_out) != {len(outputs)}:\n"
                                                              f"        raise TypeError('{function} must return a tuple of {len(outputs)} values')\n" if len(outputs) > 1 else "")
                  + f"    return {ret}\n")
        src_info = self.resolve_info(repo_id, commit)
        tree = self.tree(repo_id, commit)
        record = {"format": "void-repo-import/1", "url": info["url"], "commit": commit, "tree": src_info["tree"], "path": path, "blob": info["blob"],
                  "fileSha256": info["sha256"], "function": function, "parameters": params, "imports": info["imports"],
                  "modules": [{k: b[k] for k in ("module", "path", "blob", "sha256", "bytes")} for b in bundle],
                  "license": tree["license"], "declaredDependencies": tree["dependencies"]["declared"], "dependencyNotes": tree["dependencies"]["notes"],
                  "pins": sorted(pins or []), "interface": interface, "sourceSha256": _sha256(source.encode())}
        data = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        import_id = _sha256(data)
        dest = self.imports / f"{import_id}.json"
        if not dest.exists():
            tmp = dest.with_suffix(".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, dest)
        origin = {"importId": import_id, "url": info["url"], "commit": commit, "path": path, "blob": info["blob"], "function": function,
                  "modules": [{"path": b["path"], "blob": b["blob"]} for b in bundle],
                  "importedSourceSha256": record["sourceSha256"]}
        block = {**interface, "source": source, "dependencies": sorted(pins or []), "origin": origin}
        return {"importId": import_id, "record": record, "block": block}

    def resolve_info(self, repo_id: str, commit: str) -> dict[str, str]:
        m, _ = self._mirror_by_id(repo_id)
        return {"tree": self._git(m, "show", "-s", "--format=%T", self._commit(m, commit)).decode().strip()}

    def get_import(self, import_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[0-9a-f]{64}", import_id):
            raise RepoError("E_REPO_IMPORT", "Unknown import id.", 404)
        p = self.imports / f"{import_id}.json"
        if not p.exists():
            raise RepoError("E_REPO_IMPORT", "Unknown import id.", 404)
        data = p.read_bytes()
        if _sha256(data) != import_id:
            raise RepoError("E_REPO_IMPORT_INTEGRITY", "The import record does not match its identity.", 409)
        return json.loads(data)

    # ------------------------------------------------------------------ mirror retention
    def mirrors(self) -> list[dict[str, Any]]:
        out = []
        imports = self.list_imports()
        for m in sorted(self.root.glob("*.git")):
            meta = m / "void-source.json"
            if not meta.exists():
                continue
            url = json.loads(meta.read_text())["url"]
            size = sum(f.stat().st_size for f in m.rglob("*") if f.is_file())
            out.append({"repoId": m.name[:-4], "url": url, "bytes": size, "importsFromUrl": sum(i["url"] == url for i in imports),
                        "lastFetched": (m / "FETCH_HEAD").stat().st_mtime if (m / "FETCH_HEAD").exists() else None})
        return out

    def remove_mirror(self, repo_id: str) -> dict[str, Any]:
        """Delete a bare mirror. Import records and imported blocks keep their pinned identity and source text; browsing, compare and
        re-import need a fresh resolve() afterwards."""
        import shutil

        m, url = self._mirror_by_id(repo_id)
        size = sum(f.stat().st_size for f in m.rglob("*") if f.is_file())
        shutil.rmtree(m)
        return {"repoId": repo_id, "url": url, "bytesFreed": size, "importRecordsKept": sum(i["url"] == url for i in self.list_imports())}

    def list_imports(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.imports.glob("*.json")):
            r = json.loads(p.read_text())
            out.append({"importId": p.stem, **{k: r[k] for k in ("url", "commit", "path", "function", "pins", "sourceSha256")}})
        return out


def origin_status(block: dict[str, Any]) -> dict[str, Any] | None:
    """Is a code block still the unmodified pinned import? Compares its current source with the generated source hash."""
    o = block.get("origin")
    if not o:
        return None
    cur = _sha256(block.get("source", "").encode())
    return {**o, "currentSourceSha256": cur, "locallyModified": cur != o.get("importedSourceSha256")}


def _literal(node: ast.AST) -> bool:
    try:
        ast.literal_eval(node)
        return True
    except (ValueError, SyntaxError, TypeError):
        return False


def _req(spec: str, where: str) -> dict[str, Any]:
    spec = spec.strip()
    m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?\s*(.*)$", spec)
    name, rest = (m.group(1), m.group(3).split(";")[0].strip()) if m else (spec, "")
    pinned = bool(re.fullmatch(r"==\s*[A-Za-z0-9.+!_-]+", rest))
    return {"spec": spec, "name": name, "constraint": rest or None, "pinned": pinned, "pin": f"{name}=={rest[2:].strip()}" if pinned else None, "source": where}


def _module_name(path: str) -> str:
    p = path[:-3]
    return (p[:-len("/__init__")] if p.endswith("/__init__") else p).replace("/", ".")


def _module_index(entries: list[dict[str, Any]]) -> dict[str, str]:
    """Dotted module name -> repository path for every importable Python file (symlinks never followed)."""
    out = {}
    for e in entries:
        p = e["path"]
        if e["type"] == "blob" and e["mode"] != "120000" and p.endswith(".py") and all(part.isidentifier() for part in p[:-3].split("/")):
            out[_module_name(p)] = p
    return out


def _bundle_preamble(bundle: list[dict[str, Any]], entry_module: str, entry_is_package: bool) -> str:
    """An in-memory import hook serving the pinned module texts. Nothing is read from disk; the code still executes only in the sandbox."""
    mods = {b["module"]: (b["path"], b["package"], b["text"]) for b in bundle}
    package = entry_module if entry_is_package else entry_module.rpartition(".")[0]
    lines = ["# --- pinned repository modules (generated): served from these exact texts, never read from disk ---",
             "import sys as _void_sys, importlib.abc as _void_abc, importlib.util as _void_util",
             f"_VOID_MODULES = {mods!r}",
             "class _VoidPinnedModules(_void_abc.MetaPathFinder, _void_abc.Loader):",
             "    def find_spec(self, name, path=None, target=None):",
             "        return _void_util.spec_from_loader(name, self, is_package=_VOID_MODULES[name][1]) if name in _VOID_MODULES else None",
             "    def create_module(self, spec):",
             "        return None",
             "    def exec_module(self, module):",
             "        rel, is_pkg, text = _VOID_MODULES[module.__name__]",
             "        module.__file__ = '<pinned>/' + rel",
             "        if is_pkg:",
             "            module.__path__ = []",
             "        exec(compile(text, module.__file__, 'exec'), module.__dict__)",
             "if not any(isinstance(f, _VoidPinnedModules) for f in _void_sys.meta_path):",
             "    _void_sys.meta_path.insert(0, _VoidPinnedModules())",
             f"__package__ = {package!r}  # the entry file's own package, for its relative imports",
             "# --- entry file ---", ""]
    return "\n".join(lines) + "\n"
