"""Encrypted, authenticated single-file form of a verified offline backup.

``seal`` streams the backup directory as a tar archive through AES-256-GCM in 1 MiB
chunks. Every chunk authenticates the header, its index and whether it is the last
chunk, so truncation, reordering, splicing and header edits all fail. The key is 32
random bytes from a key file, or derived with scrypt from a passphrase file; secrets
are never accepted on the command line. ``unseal`` authenticates every chunk into a
private temporary file before extracting anything, extracts only regular files and
directories into a new destination and then runs the normal backup verification.

This protects backup confidentiality and integrity against anyone without the key.
It does not encrypt the live workbench, manage or escrow keys, or make a lost key
recoverable.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import stat
import struct
import tarfile
import tempfile

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from .core import BackupError, digest, fail, inventory, new_destination, publish, verify

MAGIC = b"VOID-SEALED-BACKUP-1\n"
SEAL_FORMAT = "void-sealed-backup-v1"
CHUNK = 1024 * 1024
SCRYPT = {"n": 2 ** 15, "r": 8, "p": 1}
MAX_HEADER = 64 * 1024
MIN_PASSPHRASE = 12


def generate_key(path) -> dict:
    """Write a new random 256-bit key file (base64 text, mode 0600). Refuses to overwrite."""
    p = Path(path).expanduser()
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(base64.b64encode(secrets.token_bytes(32)).decode() + "\n")
    return {"keyFile": str(p.resolve()), "bits": 256}


def _secret(key_file, passphrase_file):
    if (key_file is None) == (passphrase_file is None):
        fail("E_SEAL_KEY", "Provide exactly one of a key file or a passphrase file.")
    path = Path(key_file or passphrase_file).expanduser()
    mode = path.stat().st_mode
    if not stat.S_ISREG(mode):
        fail("E_SEAL_KEY", "Key/passphrase must be a regular file.")
    if mode & 0o077:
        fail("E_SEAL_KEY", "Key/passphrase file must not be readable by group or others (chmod 600).")
    raw = path.read_bytes()
    if key_file is not None:
        try:
            key = base64.b64decode(raw.strip(), validate=True)
        except ValueError:
            key = b""
        if len(key) != 32:
            fail("E_SEAL_KEY", "Key file must hold 32 bytes in base64 (see keygen).")
        return "raw", key
    phrase = raw.rstrip(b"\r\n")
    if len(phrase) < MIN_PASSPHRASE:
        fail("E_SEAL_KEY", f"Passphrase must be at least {MIN_PASSPHRASE} bytes.")
    return "scrypt", phrase


def _key(kind, secret, salt):
    if kind == "raw":
        return secret
    return Scrypt(salt=salt, length=32, **SCRYPT).derive(secret)


def _aad(header: bytes, index: int, last: bool) -> bytes:
    return header + struct.pack(">Q?", index, last)


def seal(backup, destination, *, key_file=None, passphrase_file=None, manifest_sha256=None) -> dict:
    root = Path(backup).expanduser().resolve()
    manifest = verify(root, manifest_sha256=manifest_sha256)
    manifest_sha = digest(root / "manifest.json")
    kind, secret = _secret(key_file, passphrase_file)
    dest = Path(destination).expanduser().resolve()
    if dest.exists() or dest.is_relative_to(root):
        fail("E_SEAL_DESTINATION", "Sealed file must be new and outside the backup.")
    salt, prefix = secrets.token_bytes(16), secrets.token_bytes(4)
    header = json.dumps({"format": SEAL_FORMAT, "kdf": {"name": kind, "salt": salt.hex(), **(SCRYPT if kind == "scrypt" else {})},
                         "cipher": "AES-256-GCM", "chunk": CHUNK, "noncePrefix": prefix.hex(),
                         "manifestSha256": manifest_sha, "backupFormat": manifest["format"]}, sort_keys=True).encode()
    aead = AESGCM(_key(kind, secret, salt))
    files, dirs, _ = inventory(root)
    with tempfile.TemporaryDirectory(prefix=".void-seal-", dir=dest.parent) as work:
        plain = Path(work) / "backup.tar"
        with tarfile.open(plain, "w", format=tarfile.PAX_FORMAT) as tar:
            for rel in dirs:
                tar.add(root / rel, arcname=rel, recursive=False)
            for rel, p in files.items():
                tar.add(p, arcname=rel, recursive=False)
        partial = Path(work) / "sealed"
        size, index = plain.stat().st_size, 0
        with plain.open("rb") as src, partial.open("wb") as out:
            out.write(MAGIC + struct.pack(">I", len(header)) + header)
            while True:
                chunk = src.read(CHUNK)
                last = src.tell() >= size
                nonce = prefix + struct.pack(">Q", index)
                sealed = aead.encrypt(nonce, chunk, _aad(header, index, last))
                out.write(struct.pack(">I", len(sealed)) + sealed)
                index += 1
                if last:
                    break
        os.chmod(partial, 0o600)
        os.link(partial, dest)  # refuses to replace a file created meanwhile
    return {"sealed": str(dest), "manifestSha256": manifest_sha, "kdf": kind, "chunks": index,
            "sealedSha256": digest(dest), "files": len(manifest["files"])}


def _read_exact(f, n):
    data = f.read(n)
    if len(data) != n:
        fail("E_SEAL_INTEGRITY", "Sealed backup is truncated.")
    return data


def unseal(sealed, destination, *, key_file=None, passphrase_file=None, manifest_sha256=None) -> dict:
    src_path = Path(sealed).expanduser().resolve()
    kind, secret = _secret(key_file, passphrase_file)
    with src_path.open("rb") as f:
        if f.read(len(MAGIC)) != MAGIC:
            fail("E_SEAL_FORMAT", "Not a sealed workbench backup.")
        (length,) = struct.unpack(">I", _read_exact(f, 4))
        if not 0 < length <= MAX_HEADER:
            fail("E_SEAL_FORMAT", "Invalid sealed header.")
        header = _read_exact(f, length)
        try:
            meta = json.loads(header)
            if meta["format"] != SEAL_FORMAT or meta["cipher"] != "AES-256-GCM" or meta["chunk"] != CHUNK or meta["kdf"]["name"] != kind:
                raise ValueError()
            if kind == "scrypt" and {k: meta["kdf"][k] for k in SCRYPT} != SCRYPT:
                raise ValueError()
            salt, prefix = bytes.fromhex(meta["kdf"]["salt"]), bytes.fromhex(meta["noncePrefix"])
            if len(salt) != 16 or len(prefix) != 4:
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            fail("E_SEAL_FORMAT", "Unsupported sealed header or key type.")
        if manifest_sha256 is not None and meta["manifestSha256"] != manifest_sha256:
            fail("E_SEAL_MANIFEST", "Sealed manifest differs from the separately recorded SHA256.")
        aead = AESGCM(_key(kind, secret, salt))
        dest = new_destination(src_path, destination)
        work = Path(tempfile.mkdtemp(prefix=".void-unseal-", dir=dest.parent))
        try:
            os.chmod(work, 0o700)
            plain = work / "backup.tar"
            index, finished = 0, False
            with plain.open("wb") as out:
                while not finished:
                    raw = f.read(4)
                    if not raw:
                        fail("E_SEAL_INTEGRITY", "Sealed backup ends before its final chunk.")
                    if len(raw) != 4:
                        fail("E_SEAL_INTEGRITY", "Sealed backup is truncated.")
                    (n,) = struct.unpack(">I", raw)
                    if not 16 <= n <= CHUNK + 16:
                        fail("E_SEAL_INTEGRITY", "Invalid sealed chunk length.")
                    body = _read_exact(f, n)
                    nonce = prefix + struct.pack(">Q", index)
                    for last in (False, True):
                        try:
                            out.write(aead.decrypt(nonce, body, _aad(header, index, last)))
                            finished = last
                            break
                        except InvalidTag:
                            continue
                    else:
                        fail("E_SEAL_AUTH", "Wrong key or modified sealed backup.")
                    index += 1
            if f.read(1):
                fail("E_SEAL_INTEGRITY", "Unexpected bytes after the final sealed chunk.")
            # Only fully authenticated bytes reach extraction; still admit regular files/directories only.
            stage = work / "backup"
            stage.mkdir(mode=0o700)
            with tarfile.open(plain) as tar:
                members = tar.getmembers()
                for m in members:
                    p = PurePosixPath(m.name)
                    if not (m.isreg() or m.isdir()) or p.is_absolute() or ".." in p.parts or str(p) != m.name:
                        fail("E_SEAL_FORMAT", "Sealed archive holds an unsupported entry.")
                tar.extractall(stage, members=members, filter="data")
            if digest(stage / "manifest.json") != meta["manifestSha256"]:
                fail("E_SEAL_MANIFEST", "Sealed manifest differs from its authenticated header.")
            verify(stage, manifest_sha256=meta["manifestSha256"])
            publish(stage, dest)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return {"backup": str(dest), "manifestSha256": meta["manifestSha256"], "chunks": index, "authenticated": True}


__all__ = ["BackupError", "generate_key", "seal", "unseal"]
