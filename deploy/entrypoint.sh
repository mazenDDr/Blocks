#!/bin/sh
# Refuses to serve an unauthenticated control API from a container (ADR 0074). TLS when a certificate is mounted.
set -eu
if [ -z "${VOID_USERS_FILE:-}" ] && [ -z "${VOID_API_TOKEN:-}" ]; then
  echo "E_DEPLOY_AUTH: set VOID_USERS_FILE (named accounts) or VOID_API_TOKEN before exposing the control service" >&2
  exit 64
fi
set -- python -m uvicorn control.app:create_app --factory --app-dir services --host 0.0.0.0 --port "${VOID_PORT:-8000}"
if [ -n "${VOID_TLS_CERT:-}" ] || [ -n "${VOID_TLS_KEY:-}" ]; then
  [ -r "${VOID_TLS_CERT:-}" ] && [ -r "${VOID_TLS_KEY:-}" ] || { echo "E_DEPLOY_TLS: VOID_TLS_CERT and VOID_TLS_KEY must both be readable files" >&2; exit 64; }
  set -- "$@" --ssl-certfile "$VOID_TLS_CERT" --ssl-keyfile "$VOID_TLS_KEY"
fi
exec "$@"
