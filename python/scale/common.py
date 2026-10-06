from __future__ import annotations
import hashlib
import json
import re
from urllib.parse import urlsplit

class IntegrationError(Exception):
    def __init__(self, code, message):
        self.code, self.message = code, message
        super().__init__(message)

def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()

def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()

def loopback(url):
    p = urlsplit(url)
    if p.scheme != "http" or p.hostname != "127.0.0.1" or p.username or p.password or p.path not in ("", "/") or p.query or p.fragment or not p.port:
        raise IntegrationError("loopback_required", "Only http://127.0.0.1:PORT is supported. Cross-host TLS workers are not implemented.")
    return url.rstrip("/")

def worker_endpoint(url, ca_file=None):
    """Loopback HTTP (same machine) or HTTPS to another host with a pinned CA/certificate file (ADR 0071). Never cleartext across hosts."""
    p = urlsplit(url)
    if p.scheme == "http":
        if ca_file:
            raise IntegrationError("worker_tls", "A CA file applies only to https endpoints.")
        return loopback(url)
    if p.scheme != "https" or not p.hostname or p.username or p.password or p.path not in ("", "/") or p.query or p.fragment or not p.port:
        raise IntegrationError("worker_endpoint", "Use http://127.0.0.1:PORT on this machine or https://HOST:PORT for another host.")
    if not ca_file:
        raise IntegrationError("worker_tls", "An https worker needs caFile: the worker's certificate or its CA (PEM) on this machine.")
    from pathlib import Path
    if not Path(ca_file).expanduser().is_file():
        raise IntegrationError("worker_tls", f"caFile '{ca_file}' does not exist.")
    return url.rstrip("/")


def tls_verify(ca_file):
    """httpx verify argument: the pinned CA/certificate only (system roots are not trusted for workers)."""
    import ssl
    from pathlib import Path
    return True if not ca_file else ssl.create_default_context(cafile=str(Path(ca_file).expanduser()))


def secret_free(value):
    if isinstance(value, dict):
        for k, v in value.items():
            if re.search(r"password|credential|api.?key|access.?token|secret", k, re.I) and v not in (None, "", {}):
                raise IntegrationError("sharing_secret", f"Field '{k}' requires removal before export; secret sharing is unsupported.")
            secret_free(v)
    elif isinstance(value, list):
        for v in value:
            secret_free(v)
    elif isinstance(value, str) and "://" in value:
        p = urlsplit(value)
        if p.username or p.password:
            raise IntegrationError("sharing_secret", "URLs containing inline credentials cannot be exported.")
