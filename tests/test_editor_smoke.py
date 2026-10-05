"""Actual runner/process failure behavior and the native zero-model fixture contract."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest

from agent_helpers import Lab
from graph_core.schema import Graph

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('editor_smoke', ROOT / 'tools/editor_smoke.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def test_state_example_is_actual_native_thread_reducer_with_zero_model_calls(tmp_path):
    graph = Graph.model_validate(json.loads((ROOT / 'examples/serving_state.project.json').read_text()))
    lab = Lab(tmp_path)
    for i, word in enumerate(('first', 'second', 'third'), 1):
        status, rid = lab.run(graph, inp={'question': 'SYNTHETIC '+word})
        assert status == 'completed'
        assert lab.final(rid) == {'question': 'SYNTHETIC '+word, 'n': i, 'history': ['SYNTHETIC '+w for w in ('first','second','third')[:i][-2:]], 'turns': 1, 'answer': f'SYNTHETIC {word}: {i}/1'}
        assert lab.contexts(rid) == []
    assert lab.finished(rid)['modelCalls'] == 0


def test_runner_environment_does_not_inherit_user_application_or_provider_settings(monkeypatch):
    for key in ('VOID_WORKBENCH', 'VOID_API_TOKEN', 'VOID_PLUGIN_MANIFESTS', 'ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'WANDB_API_KEY', 'MLFLOW_TRACKING_URI'):
        monkeypatch.setenv(key, 'SYNTHETIC test value')
    env = smoke.isolated_env()
    assert not any(k.startswith(('VOID_', 'ANTHROPIC_', 'OPENAI_', 'WANDB_', 'MLFLOW_')) for k in env)
    assert env['PATH'] == os.environ['PATH']


def test_stop_terminates_only_owned_group_including_real_child(tmp_path):
    child_path = tmp_path / 'child.pid'
    code = "import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
    owned = subprocess.Popen([sys.executable, '-c', code, str(child_path)], start_new_session=True)
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], start_new_session=True)
    try:
        deadline = time.monotonic()+5
        while not child_path.exists() and time.monotonic()<deadline:
            time.sleep(.02)
        child = int(child_path.read_text())
        smoke.stop(owned)
        assert owned.poll() is not None
        assert unrelated.poll() is None
        deadline = time.monotonic()+5
        def child_running():
            try:
                return psutil.Process(child).status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                return False
        while time.monotonic()<deadline and child_running():
            time.sleep(.02)
        assert not child_running()
    finally:
        smoke.stop(owned)
        smoke.stop(unrelated)


def test_readiness_refuses_exited_service_and_live_timeout(tmp_path):
    stopped = subprocess.Popen([sys.executable, '-c', 'pass'], start_new_session=True)
    stopped.wait(timeout=5)
    with pytest.raises(RuntimeError, match='exited before readiness'):
        smoke.wait_ready('http://127.0.0.1:1', stopped, timeout=1)
    alive = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], start_new_session=True)
    try:
        with pytest.raises(RuntimeError, match='readiness exceeded'):
            smoke.wait_ready('http://127.0.0.1:1', alive, timeout=.1)
    finally:
        smoke.stop(alive)
