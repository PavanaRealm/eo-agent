import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent import run_agent
from tools import classify_ndvi, compute_ndvi, get_region_observations


def tool_use(id, name, **inp):
    return NS(type="tool_use", id=id, name=name, input=inp)


def text(t):
    return NS(type="text", text=t)


class FakeClient:
    """Plays back scripted model responses so the loop can be tested offline."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = NS(create=self._create)
        self.calls = []

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_ndvi_values():
    assert compute_ndvi(nir=0.38, red=0.06) == 0.7273
    assert compute_ndvi(nir=0.02, red=0.03) == -0.2


def test_ndvi_zero_division():
    with pytest.raises(ValueError):
        compute_ndvi(0, 0)


@pytest.mark.parametrize("v,expected", [
    (-0.2, "water"), (0.1, "bare"), (0.3, "sparse"), (0.8, "dense")])
def test_classes(v, expected):
    assert expected in classify_ndvi(v)


def test_unknown_region():
    with pytest.raises(ValueError):
        get_region_observations("mountain_x")


def test_agent_loop_calls_tools_then_answers():
    client = FakeClient([
        NS(stop_reason="tool_use", content=[tool_use("1", "get_region_observations", region_id="forest_a")]),
        NS(stop_reason="tool_use", content=[tool_use("2", "compute_ndvi", nir=0.38, red=0.06)]),
        NS(stop_reason="end_turn", content=[text("NDVI is 0.7273, dense vegetation.")]),
    ])
    answer, trace = run_agent("NDVI of forest_a?", client)
    assert "0.7273" in answer
    assert [s["tool"] for s in trace] == ["get_region_observations", "compute_ndvi"]
    # tool results were sent back to the model
    last_user = client.calls[-1]["messages"][-1]
    assert last_user["role"] == "user" and last_user["content"][0]["type"] == "tool_result"


def test_tool_error_is_reported_to_model():
    client = FakeClient([
        NS(stop_reason="tool_use", content=[tool_use("1", "get_region_observations", region_id="mountain_x")]),
        NS(stop_reason="end_turn", content=[text("Region mountain_x is unknown.")]),
    ])
    answer, trace = run_agent("NDVI of mountain_x?", client)
    assert "error" in trace[0]
    assert client.calls[-1]["messages"][-1]["content"][0].get("is_error") is True


def test_max_steps_stops_loop():
    looping = NS(stop_reason="tool_use", content=[tool_use("1", "classify_ndvi", ndvi=0.5)])
    client = FakeClient([looping] * 3)
    answer, _ = run_agent("loop", client, max_steps=3)
    assert answer.startswith("Stopped")
