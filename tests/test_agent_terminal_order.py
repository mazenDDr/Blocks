"""Terminal status is observable only after the actual worker's finish event exists."""
import pytest

from agent import samples as sm
from agent_helpers import Lab


@pytest.mark.parametrize('cancel', [False, True])
def test_terminal_status_has_persisted_finish_metadata(tmp_path, monkeypatch, cancel):
    lab = Lab(tmp_path)
    original = lab.store.set_status
    observed = []
    def publish(rid, status, error=None):
        if status in ('completed', 'cancelled', 'failed'):
            finished = lab.store.last_event(rid, 'run_finished')
            assert finished is not None
            assert finished['data']['status'] == status
            assert lab.store.get_run(rid)['status'] not in ('completed', 'cancelled', 'failed')
            observed.append(status)
        return original(rid, status, error)
    monkeypatch.setattr(lab.store, 'set_status', publish)
    status, rid = lab.run(sm.counter_loop_graph(stop_at=4), cancel=lambda: cancel)
    assert status == ('cancelled' if cancel else 'completed')
    assert observed == [status]
    assert lab.finished(rid)['status'] == status
    if not cancel:
        assert lab.finished(rid)['stoppedBy'] == 'end'
