"""Named accounts, roles, serving ownership and audit (ADR 0055) with SYNTHETIC users and the real control app."""
import json
import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from control.accounts import load, main
from control.app import create_app


def users(tmp_path):
    f = tmp_path / "users.json"
    tokens = {}
    for name, role in (("ana", "admin"), ("olu", "operator"), ("vic", "viewer")):
        out = subprocess.run([sys.executable, "-m", "control.accounts", "add", "--file", str(f), "--name", name, "--role", role],
                             capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "services"})
        assert out.returncode == 0, out.stderr
        tokens[name] = json.loads(out.stdout)["token"]
    return f, tokens


def h(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def app(tmp_path):
    f, tokens = users(tmp_path)
    assert oct(f.stat().st_mode & 0o777) == "0o600"
    with TestClient(create_app(tmp_path / "wb", users_file=f)) as c:
        yield c, tokens, tmp_path / "wb"


def test_file_stores_only_token_hashes_and_cli_refuses_duplicates(tmp_path):
    f, tokens = users(tmp_path)
    text = f.read_text()
    assert all(t not in text for t in tokens.values())
    assert {a.name: a.role for a in load(f).values()} == {"ana": "admin", "olu": "operator", "vic": "viewer"}
    assert main(["add", "--file", str(f), "--name", "ana", "--role", "viewer"]) == 2
    f.chmod(0o644)
    with pytest.raises(ValueError, match="chmod 600"):
        load(f)


def test_authentication_and_whoami(app):
    c, tokens, _ = app
    assert c.get("/api/whoami").status_code == 401
    assert c.get("/api/whoami", headers=h("SYNTHETIC-not-a-token-000000")).status_code == 401
    assert c.get("/api/whoami", headers=h(tokens["olu"])).json() == {"account": {"name": "olu", "role": "operator"}, "mode": "accounts"}


def test_roles_viewer_read_only_and_admin_only_operations(app):
    c, tokens, _ = app
    assert c.get("/api/production", headers=h(tokens["vic"])).status_code == 200
    r = c.post("/api/validate", headers=h(tokens["vic"]), json={"graph": {}})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "E_ROLE"
    for path in ("/api/production/releases/x/deploy", "/api/production/releases/x/rollback", "/api/production/traffic"):
        r = c.post(path, headers=h(tokens["olu"]), json={})
        assert r.status_code == 403 and "admin" in r.json()["detail"]["message"], path
    assert c.put("/api/cache/nodes/policies/p", headers=h(tokens["olu"]), json={}).status_code == 403
    assert c.post("/api/connections", headers=h(tokens["olu"]), json={}).status_code == 403
    # The admin passes the role gate and reaches the endpoint's own validation.
    assert c.post("/api/production/releases/x/deploy", headers=h(tokens["ana"]), json={}).status_code not in (401, 403)


def test_serving_ownership_and_filtered_request_listing(app):
    c, tokens, _ = app
    r = c.post("/api/serve/local/none/predict", headers=h(tokens["olu"]), json={"requestId": "q1", "user": "ana", "records": [{}]})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "E_OWNER"
    r = c.post("/api/serve/local/none/predict/stream", headers=h(tokens["olu"]), json={"requestId": "q1", "user": "ana", "records": [{}]})
    assert r.status_code == 403
    for path in ("/api/production/requests/q1?user=ana", "/api/production/releases/r/conversations?user=ana",
                 "/api/production/releases/r/conversation?user=ana&session=s"):
        assert c.get(path, headers=h(tokens["olu"])).json()["detail"]["code"] == "E_OWNER", path
    for path in ("/api/production/requests/q1/cancel", "/api/production/requests/q1/labels", "/api/production/requests/q1/replay"):
        body = {"user": "ana", **({"labels": [1]} if path.endswith("labels") else {})}
        assert c.post(path, headers=h(tokens["olu"]), json=body).json()["detail"]["code"] == "E_OWNER", path
    # Own namespace passes ownership (and then fails honestly on the missing route); admin may act for others.
    own = c.post("/api/serve/local/none/predict", headers=h(tokens["olu"]), json={"requestId": "q2", "user": "olu", "records": [{}]})
    assert own.status_code != 403
    assert c.get("/api/production/requests/q1?user=olu", headers=h(tokens["ana"])).status_code != 403
    # Listing filters stored traces by owner for non-admins (store read patched with two SYNTHETIC rows).
    store = c.app.state.services.production.ps
    rows = [{"requestId": "a", "user": "olu"}, {"requestId": "b", "user": "ana"}]
    original, store.traces = store.traces, (lambda release=None: rows)
    try:
        assert [t["requestId"] for t in c.get("/api/production/requests", headers=h(tokens["olu"])).json()["requests"]] == ["a"]
        assert len(c.get("/api/production/requests", headers=h(tokens["ana"])).json()["requests"]) == 2
    finally:
        store.traces = original


def test_audit_records_mutations_without_tokens(app):
    c, tokens, wb = app
    c.get("/api/whoami", headers=h(tokens["vic"]))
    c.post("/api/validate", headers=h(tokens["vic"]), json={"graph": {}})
    c.post("/api/validate", headers=h(tokens["olu"]), json={"graph": {}})
    c.post("/api/validate", json={"graph": {}})
    lines = [json.loads(x) for x in (wb / "audit/requests.jsonl").read_text().splitlines()]
    assert [(x["account"], x["status"]) for x in lines] == [("vic", 403), ("olu", lines[1]["status"]), (None, 401)]
    assert all(x["path"] == "/api/validate" and x["method"] == "POST" for x in lines)
    text = (wb / "audit/requests.jsonl").read_text()
    assert all(t not in text for t in tokens.values())
    assert oct((wb / "audit/requests.jsonl").stat().st_mode & 0o777) == "0o600"


def test_shared_token_mode_is_unchanged_and_modes_are_exclusive(tmp_path):
    f, _ = users(tmp_path)
    with pytest.raises(ValueError, match="not both"):
        create_app(tmp_path / "wb", api_token="SYNTHETIC-shared-token-123456", users_file=f)
    with TestClient(create_app(tmp_path / "wb2", api_token="SYNTHETIC-shared-token-123456")) as c:
        assert c.get("/api/whoami").status_code == 401
        assert c.get("/api/whoami", headers=h("SYNTHETIC-shared-token-123456")).json() == {"account": None, "mode": "shared-token"}
    with TestClient(create_app(tmp_path / "wb3")) as c:
        assert c.get("/api/whoami").json() == {"account": None, "mode": "open"}


def test_real_uvicorn_tls_with_named_accounts(tmp_path):
    """The documented launch: uvicorn's own TLS with a SYNTHETIC self-signed certificate, verified by the client."""
    import datetime
    import socket
    import ssl
    import time

    import httpx
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=1)).not_valid_after(now + datetime.timedelta(hours=1))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True).sign(key, hashes.SHA256()))
    (tmp_path / "cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (tmp_path / "key.pem").write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    (tmp_path / "key.pem").chmod(0o600)
    f, tokens = users(tmp_path)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "VOID_WORKBENCH": str(tmp_path / "wb"), "VOID_USERS_FILE": str(f)}
    env.pop("VOID_API_TOKEN", None)
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "control.app:create_app", "--factory", "--app-dir", "services", "--host", "127.0.0.1",
                             "--port", str(port), "--ssl-certfile", str(tmp_path / "cert.pem"), "--ssl-keyfile", str(tmp_path / "key.pem")],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        url = f"https://localhost:{port}/api/whoami"
        with httpx.Client(verify=ssl.create_default_context(cafile=str(tmp_path / "cert.pem")), trust_env=False) as client:
            for _ in range(150):
                try:
                    r = client.get(url, headers=h(tokens["vic"]))
                    break
                except httpx.ConnectError:
                    assert proc.poll() is None, proc.stderr.read().decode()[-2000:]
                    time.sleep(0.1)
            assert r.status_code == 200 and r.json()["account"] == {"name": "vic", "role": "viewer"}
            assert client.get(url).status_code == 401
        with httpx.Client(trust_env=False) as plain:  # system trust store: the self-signed certificate is refused
            with pytest.raises(httpx.ConnectError):
                plain.get(url, headers=h(tokens["vic"]))
        with httpx.Client(trust_env=False) as cleartext:  # no cleartext HTTP on the TLS port
            with pytest.raises(httpx.HTTPError):
                cleartext.get(f"http://localhost:{port}/api/whoami", headers=h(tokens["vic"]))
    finally:
        proc.terminate()
        proc.wait(timeout=10)
