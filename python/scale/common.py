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
