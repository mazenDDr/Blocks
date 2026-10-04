"""A44: import repository code. A real local Git repository is browsed at pinned commits without executing anything (installation scripts,
conftest, hooks and diff/filter drivers leave no marker), dependencies/license/large-file pointers/submodules are reported statically, and a
selected function is imported as a code block whose commit/path/blob/dependency pins are recorded immutably and run only in the sandbox."""
import json
import os
import subprocess
import uuid

import pytest
import torch
from fastapi.testclient import TestClient

from codeblocks.testing import check_block
from control.app import create_app
from graph_core.hashing import semantic_hash
from graph_core.schema import CodeBlockDef, Graph
from repos import RepoError, Repos, check_url, origin_status

STATS_V1 = '''"""Small statistics helpers."""
import torch


def standardize(x, eps=1e-6):
    """Column-standardize a 2-D tensor (population std)."""
    mean = x.mean(0)
    return (x - mean) / (x.std(0, unbiased=False) + eps)


def summary(x):
    return x.mean(0), x.std(0, unbiased=False)
'''
STATS_V2 = STATS_V1.replace("unbiased=False) + eps)", "unbiased=True) + eps)")
LFS = "version https://git-lfs.github.com/spec/v1\noid sha256:" + "a" * 64 + "\nsize 12345\n"
MIT = "MIT License\n\nCopyright (c) 2026 Fixture\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\n"


def git(cwd, *args):
    env = {**os.environ, "GIT_AUTHOR_NAME": "Fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid", "GIT_COMMITTER_NAME": "Fixture",
           "GIT_COMMITTER_EMAIL": "fixture@example.invalid", "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z"}
    return subprocess.run(["git", "-c", "init.defaultBranch=main", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def source(tmp_path):
    """A repository whose installation/test/hook/driver code would each write a marker file if anything executed it."""
    marker = tmp_path / "EXECUTED"
    src = tmp_path / "src"
    src.mkdir()
    git(src, "init", "-q")
    trap = f"open({str(marker)!r}, 'a').write(__file__ + '\\n')\n"
    files = {
        "stats_utils.py": STATS_V1,
        "multi.py": "from pkg.helpers import double\n\ndef twice(x):\n    return double(x)\n",
        "pkg/__init__.py": trap, "pkg/helpers.py": "def double(x):\n    return 2 * x\n",
        "setup.py": "import setuptools\n" + trap + "setuptools.setup(name='fixture', install_requires=['requests'])\n",
        "conftest.py": trap, "install.sh": f"#!/bin/sh\ntouch {marker}\n",
        "requirements.txt": "numpy==2.5.3\ntorch>=2.0  # any recent\n-e .\n",
        "pyproject.toml": '[project]\nname = "fixture"\ndependencies = ["scipy==1.18.1"]\n[project.optional-dependencies]\nplot = ["matplotlib"]\n'
                          '[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"\n',
        "LICENSE": MIT, "data/big.parquet": LFS, "data/table.csv": "a,b\n1,2\n", "config/train.yaml": "lr: 0.1\n",
        ".gitattributes": "*.py diff=trap filter=trap\n",
    }
    for p, t in files.items():
        (src / p).parent.mkdir(parents=True, exist_ok=True)
        (src / p).write_text(t)
    # drivers and hooks live in the SOURCE repository's local config; a fetch must never carry or run them
    git(src, "config", "diff.trap.command", f"touch {marker}")
    git(src, "config", "filter.trap.smudge", f"touch {marker}; cat")
    hook = src / ".git/hooks/post-checkout"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\n")
    hook.chmod(0o755)
    git(src, "add", "-A")
    git(src, "update-index", "--add", "--cacheinfo", "160000," + "b" * 40 + ",vendor/lib")  # a submodule gitlink
    git(src, "commit", "-q", "-m", "v1")
    git(src, "tag", "v1")
    c1 = git(src, "rev-parse", "HEAD")
    (src / "stats_utils.py").write_text(STATS_V2)
    git(src, "commit", "-q", "-am", "use the sample std")
    git(src, "tag", "v2")
    c2 = git(src, "rev-parse", "HEAD")
    return {"path": str(src), "marker": marker, "c1": c1, "c2": c2}


@pytest.fixture
def repos(tmp_path):
    return Repos(tmp_path / "wb")


INTERFACE = {"id": "standardize", "description": "imported", "inputs": [{"name": "x", "shape": ["N", 3]}], "outputs": [{"name": "y", "same_as": "x"}],
             "config": [{"name": "eps", "type": "float", "default": 1e-6}]}


def test_resolve_pins_branches_tags_and_shas_without_a_checkout(repos, source):
    head = repos.resolve(source["path"])
    assert head["commit"] == source["c2"] and len(head["tree"]) == 40 and "nothing was checked out" in head["note"]
    assert repos.resolve(source["path"], "v1")["commit"] == source["c1"]
    assert repos.resolve(source["path"], "main")["commit"] == source["c2"]
    assert repos.resolve(source["path"], source["c1"])["commit"] == source["c1"]
    with pytest.raises(RepoError, match="not found"):
        repos.resolve(source["path"], "no-such-branch")
    mirror = repos.mirror_of(source["path"])
    assert (mirror / "HEAD").exists() and not any(p.name == "stats_utils.py" for p in repos.root.rglob("*"))  # bare: no working tree
    assert not source["marker"].exists()


@pytest.mark.parametrize("url,code", [("https://user:token@example.com/r.git", "E_REPO_CREDENTIALS"), ("ext::sh -c touch% /tmp/x", "E_REPO_URL"),
                                      ("relative/path", "E_REPO_URL"), ("ftp://example.com/r.git", "E_REPO_URL"), ("--upload-pack=touch", "E_REPO_URL")])
def test_unsafe_urls_are_refused(url, code):
    with pytest.raises(RepoError) as e:
        check_url(url)
    assert e.value.code == code


def test_tree_classifies_content_and_reads_dependencies_and_license_statically(repos, source):
    r = repos.resolve(source["path"], "v1")
    t = repos.tree(r["repoId"], r["commit"])
    kind = {e["path"]: e["kind"] for e in t["entries"]}
    assert kind["setup.py"] == kind["conftest.py"] == kind["install.sh"] == "installation_script"
    assert kind["data/big.parquet"] == "lfs_pointer" and kind["vendor/lib"] == "submodule" and kind["data/table.csv"] == "dataset"
    assert kind["config/train.yaml"] == "configuration" and kind["stats_utils.py"] == "python_source" and kind["requirements.txt"] == "dependency_manifest"
    deps = {d["name"]: d for d in t["dependencies"]["declared"]}
    assert deps["numpy"]["pin"] == "numpy==2.5.3" and deps["torch"]["pinned"] is False and deps["scipy"]["pinned"] is True and "matplotlib" in deps
    assert "requests" not in deps  # computed by setup.py, which is never run
    notes = " ".join(t["dependencies"]["notes"])
    assert "setup.py: executable installation script; NOT run" in notes and "-e ." in notes and "build backend" in notes
    assert t["license"]["spdxGuess"] == "MIT" and t["license"]["path"] == "LICENSE"
    f = repos.read(r["repoId"], r["commit"], "stats_utils.py")
    blob = next(e["blob"] for e in t["entries"] if e["path"] == "stats_utils.py")
    assert f["text"] == STATS_V1 and f["blob"] == blob == git(source["path"], "rev-parse", "v1:stats_utils.py")  # raw blob, no filter applied
    assert repos.read(r["repoId"], r["commit"], "data/big.parquet")["text"] == LFS  # the pointer, never a download
    with pytest.raises(RepoError):
        repos.read(r["repoId"], r["commit"], "../outside")
    with pytest.raises(RepoError):
        repos.read(r["repoId"], r["commit"], "vendor/lib")
    assert not source["marker"].exists()


def test_python_inspection_parses_without_importing(repos, source):
    r = repos.resolve(source["path"], "v1")
    info = repos.inspect_python(r["repoId"], r["commit"], "stats_utils.py")
    fns = {f["name"]: f for f in info["functions"]}
    assert set(fns) == {"standardize", "summary"} and fns["standardize"]["params"][1] == {"name": "eps", "default": 1e-6, "hasDefault": True}
    assert info["wrappable"] and info["imports"] == ["torch"]
    multi = repos.inspect_python(r["repoId"], r["commit"], "multi.py")
    assert not multi["wrappable"] and multi["localImports"] == ["pkg.helpers"]
    with pytest.raises(RepoError, match="cannot be wrapped"):
        repos.import_function(r["repoId"], r["commit"], "multi.py", "twice", {**INTERFACE, "config": []})
    repos.inspect_python(r["repoId"], r["commit"], "setup.py")  # parsed, not executed
    assert not source["marker"].exists()


def test_import_pins_identity_and_runs_only_in_the_sandbox_with_native_agreement(repos, source):
    r = repos.resolve(source["path"], "v1")
    out = repos.import_function(r["repoId"], r["commit"], "stats_utils.py", "standardize", INTERFACE, pins=["torch==" + torch.__version__])
    rec, block = out["record"], out["block"]
    assert rec["commit"] == source["c1"] and rec["path"] == "stats_utils.py" and rec["tree"] == r["tree"] and rec["license"]["spdxGuess"] == "MIT"
    assert rec["pins"] == block["dependencies"] and {d["name"] for d in rec["declaredDependencies"]} >= {"numpy", "torch", "scipy"}
    assert repos.get_import(out["importId"]) == rec and not source["marker"].exists()  # importing executes nothing either

    x = [[1.0, 2.0, 4.0], [3.0, 5.0, 9.0], [2.0, 2.0, 2.0]]
    t = torch.tensor(x)
    expect = ((t - t.mean(0)) / (t.std(0, unbiased=False) + 1e-6)).tolist()
    d = CodeBlockDef.model_validate({**block, "fixtures": [{"name": "values", "inputs": {"x": {"values": x}}, "expect": {"y": {"values": expect, "atol": 1e-6}}}]})
    res = check_block(d)
    assert res["ok"], res
    assert not source["marker"].exists()  # the imported module only ran inside the sandbox, and its own code has no trap

    # the pin is part of the semantic identity: same function at v2 gives a different import, block and graph hash
    r2 = repos.resolve(source["path"], "v2")
    out2 = repos.import_function(r2["repoId"], r2["commit"], "stats_utils.py", "standardize", INTERFACE, pins=block["dependencies"])
    assert out2["importId"] != out["importId"] and out2["block"]["origin"]["commit"] == source["c2"]
    g1, g2 = Graph(codeBlocks=[CodeBlockDef.model_validate(block)]), Graph(codeBlocks=[CodeBlockDef.model_validate(out2["block"])])
    assert g1.to_json()["codeBlocks"][0]["origin"]["commit"] == source["c1"] and semantic_hash(g1) != semantic_hash(g2)
    moved = Graph(codeBlocks=[CodeBlockDef.model_validate({**block, "origin": {**block["origin"], "commit": source["c2"]}})])
    assert semantic_hash(moved) != semantic_hash(g1)  # even the recorded pin alone changes identity


def test_local_modifications_compare_and_deliberate_update(repos, source):
    r1 = repos.resolve(source["path"], "v1")
    out = repos.import_function(r1["repoId"], r1["commit"], "stats_utils.py", "standardize", INTERFACE)
    assert origin_status(out["block"])["locallyModified"] is False
    edited = {**out["block"], "source": out["block"]["source"].replace("+ eps)", "+ 2 * eps)")}
    st = origin_status(edited)
    assert st["locallyModified"] is True and st["commit"] == source["c1"] and st["currentSourceSha256"] != st["importedSourceSha256"]
    cmp = repos.compare(r1["repoId"], source["c1"], source["c2"], ["stats_utils.py"])
    assert cmp["changed"] == [{"status": "M", "path": "stats_utils.py"}] and "-    return (x - mean) / (x.std(0, unbiased=False) + eps)" in cmp["patch"]
    assert repos.compare(r1["repoId"], source["c1"], source["c2"], ["LICENSE"])["changed"] == []
    assert not source["marker"].exists()  # diff drivers declared by .gitattributes were not run


@pytest.mark.parametrize("function,iface,match", [
    ("missing", INTERFACE, "no top-level function"),
    ("standardize", {**INTERFACE, "inputs": [{"name": "z", "shape": ["N", 3]}]}, "not parameters"),
    ("standardize", {**INTERFACE, "inputs": []}, "at least one input"),
    ("summary", {**INTERFACE, "config": []}, None),
])
def test_interface_must_match_the_function(repos, source, function, iface, match):
    r = repos.resolve(source["path"], "v1")
    if match is None:
        out = repos.import_function(r["repoId"], r["commit"], "stats_utils.py", function, {**iface, "outputs": [{"name": "m", "same_as": None, "shape": [3]}, {"name": "s", "shape": [3]}]})
        assert "_out[1]" in out["block"]["source"]
        return
    with pytest.raises(RepoError, match=match):
        repos.import_function(r["repoId"], r["commit"], "stats_utils.py", function, iface)


def test_tampered_import_record_is_refused(repos, source):
    r = repos.resolve(source["path"], "v1")
    out = repos.import_function(r["repoId"], r["commit"], "stats_utils.py", "standardize", INTERFACE)
    p = repos.imports / f"{out['importId']}.json"
    p.write_text(p.read_text().replace(source["c1"], source["c2"]))
    with pytest.raises(RepoError) as e:
        repos.get_import(out["importId"])
    assert e.value.code == "E_REPO_IMPORT_INTEGRITY"


def test_http_browse_and_import_journey(tmp_path, source):
    with TestClient(create_app(tmp_path / "wb")) as c:
        r = c.post("/api/repos/resolve", json={"url": source["path"], "rev": "v1"})
        assert r.status_code == 200, r.text
        rid, commit = r.json()["repoId"], r.json()["commit"]
        tree = c.get(f"/api/repos/{rid}/tree", params={"commit": commit}).json()
        assert tree["counts"]["installation_script"] == 3 and tree["license"]["spdxGuess"] == "MIT"
        assert c.get(f"/api/repos/{rid}/file", params={"commit": commit, "path": "stats_utils.py"}).json()["text"] == STATS_V1
        py = c.get(f"/api/repos/{rid}/python", params={"commit": commit, "path": "stats_utils.py"}).json()
        assert py["wrappable"] and [f["name"] for f in py["functions"]] == ["standardize", "summary"]
        imp = c.post(f"/api/repos/{rid}/import", json={"commit": commit, "path": "stats_utils.py", "function": "standardize", "interface": INTERFACE})
        assert imp.status_code == 201, imp.text
        iid = imp.json()["importId"]
        assert c.get(f"/api/repos/imports/{iid}").json()["commit"] == commit and any(i["importId"] == iid for i in c.get("/api/repos/imports").json()["imports"])
        assert c.post("/api/repos/origin-status", json={"block": imp.json()["block"]}).json()["origin"]["locallyModified"] is False
        cmp = c.post(f"/api/repos/{rid}/compare", json={"base": commit, "head": source["c2"], "paths": ["stats_utils.py"]}).json()
        assert cmp["changed"][0]["path"] == "stats_utils.py"
        assert c.post("/api/repos/resolve", json={"url": "https://u:p@example.com/x.git"}).json()["detail"]["code"] == "E_REPO_CREDENTIALS"
        assert c.get(f"/api/repos/{rid}/tree", params={"commit": "HEAD"}).status_code == 422
        assert c.get("/api/repos/" + "0" * 24 + "/tree", params={"commit": commit}).status_code == 404
        tested = c.post("/api/codeblocks/test", json={"block": {**imp.json()["block"], "fixtures": [{"name": "f", "inputs": {"x": {"shape": [4, 3], "seed": 1}}}]}})
        assert tested.status_code == 200 and tested.json()["ok"], tested.text
    assert not source["marker"].exists()


def test_user_git_config_cannot_reenable_hooks_or_external_diff(tmp_path, source, monkeypatch):
    """HOME is kept so the user's credential helpers work, but a user-level hooksPath (git runs reference-transaction hooks during
    fetch) or diff.external must not execute anything: the safety settings are passed with -c, which wins over user configuration."""
    home, hooks, marker = tmp_path / "home", tmp_path / "userhooks", tmp_path / "USER_HOOK_RAN"
    home.mkdir()
    hooks.mkdir()
    for name in ("reference-transaction", "post-checkout", "post-merge"):
        (hooks / name).write_text(f"#!/bin/sh\ntouch {marker}\n")
        (hooks / name).chmod(0o755)
    (home / ".gitconfig").write_text(f"[core]\n\thooksPath = {hooks}\n[diff]\n\texternal = sh -c 'touch {marker}'\n")
    monkeypatch.setenv("HOME", str(home))
    git(source["path"], "rev-parse", "HEAD")  # sanity: the user config is valid for git itself
    repos = Repos(tmp_path / "wb")
    r = repos.resolve(source["path"], "v1")
    repos.tree(r["repoId"], r["commit"])
    repos.compare(r["repoId"], source["c1"], source["c2"], ["stats_utils.py"])
    repos.resolve(source["path"], "v2")  # a second fetch updates refs again
    assert not marker.exists() and not source["marker"].exists()


def test_mirror_listing_and_removal_keep_imports(repos, source):
    r = repos.resolve(source["path"], "v1")
    out = repos.import_function(r["repoId"], r["commit"], "stats_utils.py", "standardize", INTERFACE)
    [m] = repos.mirrors()
    assert m["repoId"] == r["repoId"] and m["bytes"] > 0 and m["importsFromUrl"] == 1 and m["lastFetched"] is not None
    gone = repos.remove_mirror(r["repoId"])
    assert gone["bytesFreed"] == m["bytes"] and gone["importRecordsKept"] == 1 and repos.mirrors() == []
    assert repos.get_import(out["importId"])["commit"] == source["c1"]               # the pinned record survives
    with pytest.raises(RepoError) as e:
        repos.tree(r["repoId"], r["commit"])
    assert e.value.status == 404
    assert repos.resolve(source["path"], "v1")["commit"] == source["c1"]             # a fresh resolve restores browsing
    with pytest.raises(RepoError):
        repos.remove_mirror("../../etc")
