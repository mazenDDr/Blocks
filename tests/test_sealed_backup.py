"""Encrypted authenticated backups over real SQLite/CAS backups; SYNTHETIC keys and data only."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from artifact_store import ArtifactStore
from workbench_backup import core as backup
from workbench_backup.sealed import CHUNK, MAGIC, generate_key, seal, unseal


def made_backup(tmp_path, payload=b"SYNTHETIC persisted bytes"):
    store = ArtifactStore(tmp_path / "source")
    store.create_run("finished", "semantic-graph-hash", {})
    for state in ("preparing", "running", "completed"):
        store.set_status("finished", state)
    store.add_artifact("finished", "SYNTHETIC_fixture", payload, "complete", None, {})
    (store.root / "projects").mkdir()
    (store.root / "projects/demo.json").write_text(json.dumps({"synthetic": True}))
    (store.root / "empty").mkdir()
    result = backup.create(store.root, tmp_path / "backup", offline=True)
    return tmp_path / "backup", result["manifestSha256"]


def passphrase(tmp_path, text=b"SYNTHETIC correct horse battery\n"):
    p = tmp_path / "phrase"
    p.write_bytes(text)
    p.chmod(0o600)
    return p


def expect(code, fn):
    with pytest.raises(backup.BackupError) as e:
        fn()
    assert e.value.code == code, str(e.value)


def test_key_file_roundtrip_restores_identical_verified_backup(tmp_path):
    bk, sha = made_backup(tmp_path)
    key = generate_key(tmp_path / "key")["keyFile"]
    assert os.stat(key).st_mode & 0o777 == 0o600
    sealed = seal(bk, tmp_path / "b.sealed", key_file=key, manifest_sha256=sha)
    assert sealed["kdf"] == "raw" and sealed["manifestSha256"] == sha
    raw = (tmp_path / "b.sealed").read_bytes()
    assert raw.startswith(MAGIC) and b"SYNTHETIC persisted bytes" not in raw and b"semantic-graph-hash" not in raw
    out = unseal(tmp_path / "b.sealed", tmp_path / "unsealed", key_file=key, manifest_sha256=sha)
    assert out["authenticated"] and out["manifestSha256"] == sha
    assert backup.digest(tmp_path / "unsealed/manifest.json") == sha
    restored = backup.restore(tmp_path / "unsealed", tmp_path / "recovered", trusted=True, manifest_sha256=sha)
    assert restored["files"] == len(json.loads((bk / "manifest.json").read_text())["files"])
    assert ArtifactStore(tmp_path / "recovered").get_run("finished")["status"] == "completed"


def test_passphrase_roundtrip_and_multi_chunk_payload(tmp_path):
    bk, sha = made_backup(tmp_path, payload=os.urandom(CHUNK * 2 + 123))
    phrase = passphrase(tmp_path)
    sealed = seal(bk, tmp_path / "b.sealed", passphrase_file=phrase)
    assert sealed["kdf"] == "scrypt" and sealed["chunks"] >= 3
    unseal(tmp_path / "b.sealed", tmp_path / "unsealed", passphrase_file=phrase, manifest_sha256=sha)
    backup.verify(tmp_path / "unsealed", manifest_sha256=sha)


def test_wrong_key_wrong_passphrase_and_wrong_key_type_refuse_without_output(tmp_path):
    bk, _ = made_backup(tmp_path)
    key = generate_key(tmp_path / "key")["keyFile"]
    other = generate_key(tmp_path / "other")["keyFile"]
    seal(bk, tmp_path / "k.sealed", key_file=key)
    expect("E_SEAL_AUTH", lambda: unseal(tmp_path / "k.sealed", tmp_path / "a", key_file=other))
    expect("E_SEAL_FORMAT", lambda: unseal(tmp_path / "k.sealed", tmp_path / "b", passphrase_file=passphrase(tmp_path)))
    seal(bk, tmp_path / "p.sealed", passphrase_file=passphrase(tmp_path))
    wrong = tmp_path / "wrong"
    wrong.write_bytes(b"SYNTHETIC incorrect phrase")
    wrong.chmod(0o600)
    expect("E_SEAL_AUTH", lambda: unseal(tmp_path / "p.sealed", tmp_path / "c", passphrase_file=wrong))
    assert not any((tmp_path / n).exists() for n in "abc")
    assert not list(tmp_path.glob(".void-unseal-*"))


def test_tampering_truncation_reordering_and_trailing_bytes_are_detected(tmp_path):
    bk, _ = made_backup(tmp_path, payload=os.urandom(CHUNK * 2))
    key = generate_key(tmp_path / "key")["keyFile"]
    seal(bk, tmp_path / "b.sealed", key_file=key)
    good = (tmp_path / "b.sealed").read_bytes()
    (hlen,) = struct.unpack(">I", good[len(MAGIC):len(MAGIC) + 4])
    start = len(MAGIC) + 4 + hlen
    chunks, at = [], start
    while at < len(good):
        (n,) = struct.unpack(">I", good[at:at + 4])
        chunks.append(good[at:at + 4 + n])
        at += 4 + n
    variants = {
        "flip": good[:-5] + bytes([good[-5] ^ 1]) + good[-4:],
        "header": good[:len(MAGIC) + 10] + bytes([good[len(MAGIC) + 10] ^ 1]) + good[len(MAGIC) + 11:],
        "truncated": good[:start] + b"".join(chunks[:-1]),
        "reordered": good[:start] + chunks[1] + chunks[0] + b"".join(chunks[2:]),
        "trailing": good + b"\x00",
        "cut": good[:-3],
    }
    for name, data in variants.items():
        p = tmp_path / f"{name}.sealed"
        p.write_bytes(data)
        with pytest.raises(backup.BackupError) as e:
            unseal(p, tmp_path / f"out-{name}", key_file=key)
        assert e.value.code in {"E_SEAL_AUTH", "E_SEAL_INTEGRITY", "E_SEAL_FORMAT"}, (name, e.value.code)
        assert not (tmp_path / f"out-{name}").exists(), name


def test_key_hygiene_destination_and_recorded_manifest_checks(tmp_path):
    bk, sha = made_backup(tmp_path)
    loose = tmp_path / "loose"
    loose.write_text("x" * 44)
    loose.chmod(0o644)
    expect("E_SEAL_KEY", lambda: seal(bk, tmp_path / "x.sealed", key_file=loose))
    short = tmp_path / "short"
    short.write_bytes(b"short")
    short.chmod(0o600)
    expect("E_SEAL_KEY", lambda: seal(bk, tmp_path / "x.sealed", passphrase_file=short))
    key = generate_key(tmp_path / "key")["keyFile"]
    with pytest.raises(FileExistsError):
        generate_key(tmp_path / "key")
    expect("E_SEAL_KEY", lambda: seal(bk, tmp_path / "x.sealed", key_file=key, passphrase_file=key))
    expect("E_SEAL_DESTINATION", lambda: seal(bk, bk / "inside.sealed", key_file=key))
    expect("E_BACKUP_MANIFEST", lambda: seal(bk, tmp_path / "x.sealed", key_file=key, manifest_sha256="0" * 64))
    seal(bk, tmp_path / "b.sealed", key_file=key)
    expect("E_SEAL_MANIFEST", lambda: unseal(tmp_path / "b.sealed", tmp_path / "u", key_file=key, manifest_sha256="0" * 64))
    (tmp_path / "exists").mkdir()
    expect("E_BACKUP_DESTINATION", lambda: unseal(tmp_path / "b.sealed", tmp_path / "exists", key_file=key))
    expect("E_SEAL_DESTINATION", lambda: seal(bk, tmp_path / "b.sealed", key_file=key))


def test_cli_keygen_seal_unseal(tmp_path):
    bk, sha = made_backup(tmp_path)
    run = lambda *a: subprocess.run([sys.executable, "-m", "workbench_backup", *a], capture_output=True, text=True)
    assert run("keygen", str(tmp_path / "key")).returncode == 0
    r = run("seal", str(bk), str(tmp_path / "b.sealed"), "--key-file", str(tmp_path / "key"), "--manifest-sha256", sha)
    assert r.returncode == 0, r.stderr
    r = run("unseal", str(tmp_path / "b.sealed"), str(tmp_path / "u"), "--key-file", str(tmp_path / "key"), "--manifest-sha256", sha)
    assert r.returncode == 0 and json.loads(r.stdout)["authenticated"], r.stderr
    r = run("unseal", str(tmp_path / "b.sealed"), str(tmp_path / "v"))
    assert r.returncode == 1 and "E_SEAL_KEY" in r.stderr
