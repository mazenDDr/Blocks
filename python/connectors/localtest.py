"""LOCAL TEST SERVICES - not production infrastructure.

Real engines started on this machine for tests and the connected-data example:
  * PostgreSQL 16 via the `pgserver` pip package (embedded real postgres binaries, Unix-domain socket, trust auth)
  * an S3-compatible server via `moto[server]` (a mock of the S3 API, not AWS; it does not enforce IAM)
  * a Git + DVC repository with a local DVC remote built with the `dvc` CLI
The connectors themselves are generic; only these helpers point them at local services."""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


class LocalPostgres:
    def __init__(self) -> None:
        import pgserver

        # a Unix-socket path must stay under ~100 characters, so use a short directory
        self.dir = tempfile.mkdtemp(prefix="vpg", dir="/tmp" if os.path.isdir("/tmp") else None)
        self.server = pgserver.get_server(self.dir, cleanup_mode="stop")
        self.host, self.port, self.dbname = self.dir, 5432, "postgres"
        self.admin_uri = self.server.get_uri()

    def admin(self):
        import psycopg

        return psycopg.connect(self.admin_uri, autocommit=True)

    def stop(self) -> None:
        try:
            self.server.cleanup()
        finally:
            shutil.rmtree(self.dir, ignore_errors=True)

    def settings(self, user: str = "postgres", **kw: Any) -> dict[str, Any]:
        return {"host": self.host, "port": self.port, "dbname": self.dbname, "user": user, "sslmode": "disable", **kw}


class LocalS3:
    ACCESS, SECRET = "local-test-access-key", "local-test-secret-key-0123456789"

    def __init__(self) -> None:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        self.endpoint = f"http://127.0.0.1:{self.port}"
        self.proc = subprocess.Popen([sys.executable, "-m", "moto.server", "-H", "127.0.0.1", "-p", str(self.port)],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        end = time.time() + 30
        while time.time() < end:
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=0.5).close()
                break
            except OSError:
                time.sleep(0.2)
        else:
            self.stop()
            raise RuntimeError("moto server did not start")

    def client(self):
        import boto3

        return boto3.client("s3", endpoint_url=self.endpoint, region_name="us-east-1", aws_access_key_id=self.ACCESS, aws_secret_access_key=self.SECRET)

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def make_dvc_repo(root: str | Path, revisions: list[tuple[str, str]], file_name: str = "data.csv") -> dict[str, Any]:
    """Create <root>/repo (git + dvc) and <root>/remote (local DVC remote); commit one revision per (tag, csv text); push to the remote."""
    root = Path(root)
    repo, remote = root / "repo", root / "remote"
    repo.mkdir(parents=True)
    remote.mkdir(parents=True)
    dvc = [str(Path(sys.executable).parent / "dvc")]
    env = {**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@example.invalid",
           "DVC_NO_ANALYTICS": "1", "GIT_CONFIG_NOSYSTEM": "1"}

    def sh(*a: str) -> str:
        return subprocess.run(list(a), cwd=repo, check=True, capture_output=True, text=True, env=env).stdout

    sh("git", "init", "-q", "-b", "main")
    sh(*dvc, "init", "-q")
    sh(*dvc, "remote", "add", "-d", "store", str(remote))
    commits = {}
    for tag, text in revisions:
        (repo / file_name).write_text(text)
        sh(*dvc, "add", file_name)
        sh("git", "add", "-A")
        sh("git", "commit", "-qm", f"data {tag}")
        sh("git", "tag", tag)
        sh(*dvc, "push")
        commits[tag] = sh("git", "rev-parse", "HEAD").strip()
    shutil.rmtree(repo / ".dvc" / "cache", ignore_errors=True)  # force materialization from the remote
    return {"repo": str(repo), "remote": str(remote), "commits": commits, "file": file_name}
