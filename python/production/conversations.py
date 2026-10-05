"""Reviewed native reset/fork/historical restore, serialized with production turns."""
from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
import uuid

from langgraph.checkpoint.memory import InMemorySaver

from tabular.core import dumps
from .pipeline import ProductionError, read_verified


def action(runtime, release_id, operation, req):
    if operation not in ("reset", "fork", "restore"):
        raise ProductionError("E_SESSION_ACTION", "Unsupported conversation action.")
    ps = runtime.ps
    release = ps.get('release', release_id)
    if release['config']['sessionMode'] != 'conversation':
        raise ProductionError('E_RELEASE_CONFIG', 'This release has no native conversations.')
    data = {'releaseId': release_id, 'operation': operation, **req.model_dump()}
    fingerprint = hashlib.sha256(dumps(data).encode()).hexdigest()
    duplicate = ps.conversation_action(req.user, req.actionId, fingerprint)
    if duplicate:
        return duplicate
    source = dumps([release_id, req.user, req.session])
    target_session = req.destinationSession if operation == 'fork' else req.session
    target = dumps([release_id, req.user, target_session])
    if operation == 'fork' and source == target:
        raise ProductionError('E_SESSION_DESTINATION', 'Fork destination must be a different, unused session.', 409)
    deadline = time.perf_counter() + release['config']['timeoutSeconds']
    pipeline = runtime.pipeline(release['versionId'])  # verify original version/provider; never execute a model
    with runtime.lock:
        locks = [runtime.session_locks.setdefault(scope, threading.Lock()) for scope in sorted({source, target})]
    acquired = []
    try:
        for lock in locks:
            if not lock.acquire(timeout=max(0, deadline-time.perf_counter())):
                raise ProductionError('E_SESSION_BUSY', 'Timed out waiting for an active turn/session action; no state changed.', 409)
            acquired.append(lock)
        duplicate = ps.conversation_action(req.user, req.actionId, fingerprint)
        if duplicate:
            return duplicate
        head = ps.conversation(source)
        if not head or (not head['checkpointSha256'] and operation != 'restore'):
            raise ProductionError('E_SESSION_EMPTY', 'Inspect a session with a committed native checkpoint first.', 409)
        if (head['revision'], head['checkpointSha256']) != (req.expectedRevision, req.expectedCheckpointSha256):
            raise ProductionError('E_SESSION_CONFLICT', 'Reviewed checkpoint changed; inspect it again before applying this action.', 409)
        # Verify decoded state with the original adapter, without changing its pinned source bytes.
        if head['checkpointSha256']:
            pipeline.checkpoint_state(head['checkpointSha256'])
        selected = head['checkpointSha256']
        if operation == 'restore':
            from .history import inspect_history
            historical = inspect_history(runtime, release_id, req.user, req.session, req.sourceRequestId, pipeline=pipeline)
            if (historical['sourceTraceSha256'], historical['sourceCheckpointSha256']) != (req.sourceTraceSha256, req.sourceCheckpointSha256):
                raise ProductionError('E_SESSION_HISTORY_CONFLICT', 'Reviewed historical source differs; inspect it again.', 409)
            selected = historical['sourceCheckpointSha256']
        checkpoint = None
        thread = None
        if operation in ('fork', 'restore'):
            if operation == 'fork' and ps.conversation(target) is not None:
                raise ProductionError('E_SESSION_DESTINATION', 'Destination session already exists; it is never overwritten.', 409)
            payload = json.loads(read_verified(runtime.store, selected))
            saver = InMemorySaver()
            native = saver.serde.loads_typed((payload['type'], base64.b64decode(payload['data'], validate=True)))
            thread = uuid.uuid4().hex
            native['metadata']['thread_id'] = thread
            typ, raw = saver.serde.dumps_typed(native)
            payload.update(threadId=thread, type=typ, data=base64.b64encode(raw).decode())
            encoded = dumps(payload).encode()
            if len(encoded) > 131_072:
                raise ProductionError('E_AGENT_CHECKPOINT_BOUNDS', 'Cloned snapshot exceeds 128 KiB; no session changed.', 413)
            checkpoint = runtime.store.put_bytes(encoded)
        return ps.commit_conversation_action(data, fingerprint, source, target, head, checkpoint, thread, deadline)
    finally:
        for lock in reversed(acquired):
            lock.release()
