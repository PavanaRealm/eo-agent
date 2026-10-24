import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).parent.parent))
import agent_gemini
from agent_gemini import _config, run_agent


def call(name, **args):
    return NS(name=name, args=args)


def reply(calls=None, text=None):
    return NS(function_calls=calls, text=text, candidates=[NS(content=NS(role="model", parts=[]))])


class FakeClient:
    """Plays back scripted model responses so the loop can be tested offline."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.models = NS(generate_content=self._gen)

    def _gen(self, **kwargs):
        self.calls.append(kwargs)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_config_builds_with_real_sdk_types():
    config = _config()
    names = [d.name for d in config.tools[0].function_declarations]
    assert names == ["get_region_observations", "compute_ndvi", "classify_ndvi"]
    assert config.automatic_function_calling.disable is True


def test_loop_calls_tools_then_answers():
    client = FakeClient([
        reply(calls=[call("get_region_observations", region_id="forest_a")]),
        reply(calls=[call("compute_ndvi", nir=0.38, red=0.06)]),
        reply(text="NDVI is 0.7273, dense vegetation."),
    ])
    answer, trace = run_agent("NDVI of forest_a?", client, model="x")
    assert "0.7273" in answer
    assert [s["tool"] for s in trace] == ["get_region_observations", "compute_ndvi"]
    sent = client.calls[-1]["contents"][-1]  # tool results go back to the model
    assert sent.role == "user"
    assert sent.parts[0].function_response.response == {"result": 0.7273}


def test_tool_error_is_reported_to_model():
    client = FakeClient([
        reply(calls=[call("get_region_observations", region_id="mountain_x")]),
        reply(text="Region mountain_x is unknown."),
    ])
    answer, trace = run_agent("NDVI of mountain_x?", client, model="x")
    assert "error" in trace[0]
    sent = client.calls[-1]["contents"][-1]
    assert "error" in sent.parts[0].function_response.response


def test_max_steps_stops_loop():
    client = FakeClient([reply(calls=[call("classify_ndvi", ndvi=0.5)])] * 3)
    answer, _ = run_agent("loop", client, model="x", max_steps=3)
    assert answer.startswith("Stopped")


def test_retry_on_rate_limit(monkeypatch):
    monkeypatch.setattr(agent_gemini.time, "sleep", lambda s: None)
    limited = Exception("rate limit")
    limited.code = 429
    client = FakeClient([limited, reply(text="ok")])
    answer, _ = run_agent("q", client, model="x")
    assert answer == "ok" and len(client.calls) == 2
