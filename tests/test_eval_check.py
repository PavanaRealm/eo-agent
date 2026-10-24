import os
import sys
from pathlib import Path

os.environ["PROVIDER"] = "gemini"
sys.path.insert(0, str(Path(__file__).parent.parent))
from eval import CASES, check

CALLED = [{"tool": "compute_ndvi"}, {"tool": "classify_ndvi"}, {"tool": "get_region_observations"}]


def case(i):
    return CASES[i]


def test_rounded_number_counts_as_correct():
    answer = "forest_a NDVI is approximately 0.727, dense vegetation."
    assert check(case(0), answer, CALLED)


def test_wrong_number_fails():
    assert not check(case(0), "forest_a NDVI is 0.65, dense vegetation.", CALLED)


def test_missing_tool_call_fails():
    assert not check(case(1), "NDVI is 0.8367", [])


def test_both_numbers_required():
    assert not check(case(2), "It declined from 0.6667.", CALLED)
    assert check(case(2), "It declined from 0.6667 to 0.17.", CALLED)


def test_unknown_region_answer():
    assert check(case(5), "Region mountain_x is unknown.", CALLED)
    assert not check(case(5), "The NDVI is 0.5.", CALLED)
