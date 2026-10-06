"""Workers on other hosts over HTTPS with a pinned certificate (ADR 0071). Exercised on loopback with a real TLS worker process."""
import datetime
import ipaddress
import json
import os
import socket
import subprocess
import sys
import time

import httpx
import pytest

from scale.common import IntegrationError, worker_endpoint
from scale.remote import RemoteClient, RemoteRequest
from test_scale import recorded  # noqa: F401  (real recorded tabular run and graph)

TOKEN = "void-local-tls-token-123456789"


def certificate(directory, name):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"SYNTHETIC {name}")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=1)).not_valid_after(now + datetime.timedelta(hours=1))
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True).sign(key, hashes.SHA256()))
    (directory / f"{name}.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (directory / f"{name}.key").write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    (directory / f"{name}.key").chmod(0o600)
    return directory / f"{name}.pem", directory / f"{name}.key"


@pytest.fixture()
def tls_worker(tmp_path):
    cert, key = certificate(tmp_path, "worker")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "VOID_WORKER_TOKEN": TOKEN}
    log = (tmp_path / "worker.log").open("w")
    proc = subprocess.Popen([sys.executable, "-m", "scale.worker_server", "--workbench", str(tmp_path / "worker"), "--port", str(port),
                             "--ssl-certfile", str(cert), "--ssl-keyfile", str(key)], env=env, stdout=log, stderr=log)
    url = f"https://127.0.0.1:{port}"
    try:
        import ssl
        ctx = ssl.create_default_context(cafile=str(cert))
        for _ in range(300):
            try:
                if httpx.get(url + "/v1/health", headers={"Authorization": "Bearer " + TOKEN}, verify=ctx, timeout=1, trust_env=False).status_code == 200:
                    break
            except httpx.HTTPError:
                assert proc.poll() is None, (tmp_path / "worker.log").read_text()
                time.sleep(0.1)
        yield url, cert, proc
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        log.close()


def test_https_worker_runs_the_snapshot_with_a_pinned_certificate(recorded, tls_worker, monkeypatch):  # noqa: F811
    store, g = recorded
    url, cert, proc = tls_worker
    monkeypatch.setenv("VOID_WORKER_TOKEN", TOKEN)
    with pytest.raises(httpx.ConnectError):  # cleartext to the TLS port, or the system trust store, never works
        httpx.get(url + "/v1/health", trust_env=False, timeout=2)
    client = RemoteClient(store.root)
    r = client.submit(RemoteRequest(graph=g, endpoint=url, tokenEnv="VOID_WORKER_TOKEN", caFile=str(cert), projectId="production_sensors"))
    assert "HTTPS" in r["location"]
    end = time.time() + 60
    while time.time() < end and not r["runId"]:
        time.sleep(0.2)
        r = RemoteClient(store.root).refresh(r["id"])
    assert r["status"] == "completed", r
    native = next(a for a in store.artifacts("native", "node_summary") if a["meta"]["node"] == "metrics")
    remote = next(a for a in store.artifacts(r["runId"], "node_summary") if a["meta"]["node"] == "metrics")
    assert json.loads(store.read_artifact(native["sha256"]))["values"] == json.loads(store.read_artifact(remote["sha256"]))["values"]


def test_wrong_certificate_never_reaches_the_worker(recorded, tls_worker, tmp_path, monkeypatch):  # noqa: F811
    store, g = recorded
    url, cert, proc = tls_worker
    other, _ = certificate(tmp_path, "impostor")
    monkeypatch.setenv("VOID_WORKER_TOKEN", TOKEN)
    r = RemoteClient(store.root).submit(RemoteRequest(graph=g, endpoint=url, tokenEnv="VOID_WORKER_TOKEN", caFile=str(other), projectId="production_sensors"))
    assert not r["workerRunId"] and not r["runId"] and "CERTIFICATE_VERIFY_FAILED" in (r["error"] or "")


def test_endpoint_rules_and_worker_bind_rule(tmp_path):
    cert, _ = certificate(tmp_path, "worker")
    assert worker_endpoint("http://127.0.0.1:8778") == "http://127.0.0.1:8778"
    assert worker_endpoint("https://gpu-box.example:8443/", str(cert)) == "https://gpu-box.example:8443"
    for url, ca, code in (("http://10.0.0.5:8778", None, "loopback_required"), ("https://gpu-box:8443", None, "worker_tls"),
                          ("https://gpu-box:8443", str(tmp_path / "missing.pem"), "worker_tls"), ("http://127.0.0.1:8778", str(cert), "worker_tls"),
                          ("https://user:pw@gpu-box:8443", str(cert), "worker_endpoint")):
        with pytest.raises(IntegrationError) as e:
            worker_endpoint(url, ca)
        assert e.value.code == code, url
    out = subprocess.run([sys.executable, "-m", "scale.worker_server", "--workbench", str(tmp_path / "w"), "--host", "0.0.0.0"],
                         capture_output=True, text=True, env={**os.environ, "VOID_WORKER_TOKEN": TOKEN}, timeout=60)
    assert out.returncode == 2 and "requires TLS" in out.stderr
