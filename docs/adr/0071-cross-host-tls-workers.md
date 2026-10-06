# ADR0071: workers on other hosts over pinned-certificate HTTPS

Status: accepted (2026-10-06); loopback TLS on Mac arm64 and hosted CI, cross-host with the user's GPU machine over Tailscale (HANDOFF§72).

The separate worker (ADR0031) accepted only `http://127.0.0.1:PORT` and bound only
to loopback: a tabular job could run in another process but never on another machine,
and "cross-host TLS workers are not implemented" was the refusal text.

Decision: allow HTTPS endpoints on other hosts, with the worker's certificate pinned.

- Client: an endpoint is `http://127.0.0.1:PORT` (unchanged) or `https://HOST:PORT`
  with `caFile`, a PEM certificate/CA file on the control machine. Only that file is
  trusted (system roots are not), cleartext HTTP to another host is refused, URLs
  with credentials/paths/queries are refused, and a CA file on an http endpoint is
  refused. The job record keeps the endpoint, the CA file path and the token
  environment name (never the token).
- Worker: `--host` defaults to 127.0.0.1; any other bind address requires
  `--ssl-certfile` and `--ssl-keyfile` (refused at start-up otherwise). The bearer
  token, body limits, snapshot materialization and hash-verified artifact import
  are unchanged.
- Editor Scale workspace: optional worker certificate file field.

Cross-host evidence (`benchmarks/results/cross_host_worker_gpu_box.json`): worker on
the GPU machine (Ubuntu 24.04/WSL2, Python 3.11.14, scikit-learn 1.6.1, pandas 2.3.3,
numpy 2.3.5; project pins scikit-learn 1.9.1) bound to its Tailscale address with a
SYNTHETIC per-session self-signed certificate for that IP. From the Mac: system-trust
HTTPS refused, no-token request 401, the production_sensors tabular snapshot ran
remotely and imported in 5.0 s (20 events, 21 artifacts) with metrics identical to the
local run (accuracy 1.0, log loss 0.14975932453323396, ROC AUC 1.0). The worker was
stopped afterwards; its key and token stay in a 0700 directory on that machine.

Not provided: certificate issuance/rotation, mutual TLS, scheduling or load balancing
across several workers, GPU work on workers (tabular CPU subset only), shared storage,
or protection against a compromised worker host beyond integrity of imported bytes.

Verification: 3 pytest cases with a real TLS worker process on loopback (job completes
with native-equal metrics; system-trust and cleartext refused; a different pinned
certificate never reaches the worker; endpoint rules; non-loopback bind without TLS
refused) and the manual cross-host run above.
