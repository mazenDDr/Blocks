# ADR0074: container deployment of the control service

Status: accepted (2026-10-06); image built and run on the user's Linux GPU machine, exercised from the Mac over Tailscale (HANDOFF§76).

There was no way to deploy the control service other than running uvicorn from a
checkout; cloud deployment was listed as missing. No cloud account is available
here, so the portable unit — a container image — is what can be built and verified.

Decision: `deploy/Dockerfile` (python:3.13-slim, the pinned `python/requirements.txt`
with CPU torch, project code and examples, non-root user `void`, `/data` volume,
`VOID_WORKBENCH=/data/workbench`) and `deploy/entrypoint.sh`.

- The entrypoint refuses to start (exit 64, E_DEPLOY_AUTH) unless `VOID_USERS_FILE`
  (ADR0055 accounts) or `VOID_API_TOKEN` is set, because a container binds 0.0.0.0.
- When `VOID_TLS_CERT`/`VOID_TLS_KEY` are set, both must be readable (E_DEPLOY_TLS)
  and uvicorn serves HTTPS; otherwise HTTP for use behind a TLS-terminating proxy.
- `.dockerignore` keeps environments, node_modules, workbenches, generated data,
  docs, tests and the editor out of the build context.

Verified on the GPU machine (Docker 29.2.1, linux/amd64): image built in one pass
(5.83 GB). Start without credentials → exit 64 with E_DEPLOY_AUTH. Started with a
SYNTHETIC two-account users file and the per-session certificate, data on a host
folder, published on the Tailscale address. From the Mac: no token 401, admin
whoami, viewer write 403, system-trust HTTPS refused, a production_sensors tabular
run submitted and completed inside the container, and still present after
`docker restart` (`benchmarks/results/container_deploy_gpu_box.json`). The container
was removed afterwards; the image and the deploy folder remain on that machine.

Not provided: a registry push, Kubernetes/compose manifests for a provider, cloud
deployment, image size optimisation (all optional ML stacks are installed), GPU
images, the editor bundle (served separately by Vite in development), and automatic
certificate management.

Verification: 3 pytest cases for entrypoint refusals and Dockerfile invariants, plus
the manual build/run above.
