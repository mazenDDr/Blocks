"""Token sink path of the provider call with the scripted FIXTURE model (no language model involved)."""
import contextvars
import threading

from agent.models import TOKEN_SINK, ModelSpec, invoke_chat


def spec():
    return ModelSpec(provider="fixture", model="fixture", fixture={"responses": ["SYNTHETIC scripted reply"]})


def test_without_sink_behaviour_is_unchanged():
    out = invoke_chat(spec(), [{"role": "user", "content": "q"}], {})
    assert out["text"] == "SYNTHETIC scripted reply"


def test_sink_receives_deltas_that_concatenate_to_the_returned_text():
    seen = []
    token = TOKEN_SINK.set(seen.append)
    try:
        out = invoke_chat(spec(), [{"role": "user", "content": "q"}], {})
    finally:
        TOKEN_SINK.reset(token)
    assert seen and "".join(seen) == out["text"] == "SYNTHETIC scripted reply"
    assert out["usage"]["source"] == "unavailable"


def test_sink_is_scoped_to_its_context():
    seen, other = [], []

    def streamed():
        TOKEN_SINK.set(seen.append)
        invoke_chat(spec(), [{"role": "user", "content": "q"}], {})

    t = threading.Thread(target=contextvars.copy_context().run, args=(streamed,))
    t.start()
    t.join()
    invoke_chat(spec(), [{"role": "user", "content": "q"}], {})  # this context has no sink
    assert "".join(seen) == "SYNTHETIC scripted reply" and other == [] and TOKEN_SINK.get() is None
