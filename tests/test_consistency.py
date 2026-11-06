import os
import sys
from pathlib import Path

os.environ["PROVIDER"] = "gemini"
sys.path.insert(0, str(Path(__file__).parent.parent))
from consistency import run_once, summarize

CASES = [
    {"q": "q1", "numbers": [1.0], "must_call": "compute_ndvi"},
    {"q": "q2", "words": ["water"], "must_call": "classify_ndvi"},
]


def fake_check(case, answer, trace):
    return case["q"] in answer


def make_agent(script):
    """script: list of results per call: ('ok', tools), ('wrong', tools) or ('error', None)."""
    calls = iter(script)

    def run_agent(question, client):
        kind, tools = next(calls)
        if kind == "error":
            raise TimeoutError("504")
        answer = question if kind == "ok" else "wrong"
        return answer, [{"tool": t} for t in tools]
    return run_agent


def one_run(script):
    return run_once(None, make_agent(script), CASES, fake_check, pause=0)


def test_all_pass_is_not_flaky():
    runs = [one_run([("ok", ["compute_ndvi"]), ("ok", ["classify_ndvi"])]) for _ in range(3)]
    s = summarize(runs, CASES)
    assert s["passed_per_run"] == [2, 2, 2]
    assert s["all_cases_passed_in_every_run"]
    assert not any(c["flaky"] for c in s["cases"])


def test_sometimes_failing_case_is_flaky():
    runs = [
        one_run([("ok", ["compute_ndvi"]), ("ok", ["classify_ndvi"])]),
        one_run([("wrong", ["compute_ndvi"]), ("ok", ["classify_ndvi"])]),
        one_run([("ok", ["compute_ndvi"]), ("ok", ["classify_ndvi"])]),
    ]
    s = summarize(runs, CASES)
    assert s["cases"][0]["passes"] == 2 and s["cases"][0]["flaky"]
    assert not s["cases"][1]["flaky"]
    assert not s["all_cases_passed_in_every_run"]


def test_api_error_is_not_counted_as_wrong_answer():
    runs = [
        one_run([("error", None), ("ok", ["classify_ndvi"])]),
        one_run([("ok", ["compute_ndvi"]), ("ok", ["classify_ndvi"])]),
    ]
    s = summarize(runs, CASES)
    c0 = s["cases"][0]
    assert c0["errors"] == 1 and c0["answered"] == 1 and c0["passes"] == 1
    assert not c0["flaky"]  # the only answered run passed
    assert s["total_errors"] == 1


def test_different_tool_paths_are_counted():
    runs = [
        one_run([("ok", ["compute_ndvi"]), ("ok", ["classify_ndvi"])]),
        one_run([("ok", ["get_region_observations", "compute_ndvi"]), ("ok", ["classify_ndvi"])]),
    ]
    s = summarize(runs, CASES)
    assert len(s["cases"][0]["tool_paths"]) == 2
    assert len(s["cases"][1]["tool_paths"]) == 1
