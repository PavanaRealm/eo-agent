"""Small benchmark: does the agent use the tools and give correct answers?

Run with a real API key:  GEMINI_API_KEY=... python eval.py
(or PROVIDER=anthropic ANTHROPIC_API_KEY=... python eval.py)
Each case checks (1) the answer contains the expected numbers/words and
(2) the agent really called the required tool.
"""
import os
import re
import time

PROVIDER = os.environ.get("PROVIDER", "gemini")  # "gemini" or "anthropic"
if PROVIDER == "anthropic":
    import anthropic
    from agent import run_agent
else:
    from agent_gemini import make_client, run_agent

CASES = [
    {"q": "What is the NDVI of forest_a on 2025-07-15 and what does it mean?",
     "numbers": [0.7273], "words": ["dense"], "must_call": "compute_ndvi"},
    {"q": "What is the NDVI of forest_a on 2024-07-15?",
     "numbers": [0.8367], "must_call": "compute_ndvi"},
    {"q": "Did vegetation in crop_b decline between 2025-05-10 and 2025-09-10? Give both NDVI values.",
     "numbers": [0.6667, 0.1667], "must_call": "compute_ndvi"},
    {"q": "What land cover class is water_c on 2025-07-15?",
     "words": ["water"], "must_call": "classify_ndvi"},
    {"q": "Compare the NDVI of urban_d and forest_a on 2025-07-15.",
     "numbers": [0.1, 0.7273], "must_call": "compute_ndvi"},
    {"q": "What is the NDVI of region mountain_x?",
     "words": ["unknown", "not", "no data", "does not exist"], "any_word": True,
     "must_call": "get_region_observations"},
]

TOLERANCE = 0.005  # rounding to 2 decimals counts as correct (0.73 for 0.7273)


def numbers_in(text):
    return [float(x) for x in re.findall(r"-?\d+\.\d+|-?\d+", text)]


def check(case, answer, trace):
    text = answer.lower()
    found = numbers_in(text)
    numbers_ok = all(
        any(abs(f - want) <= TOLERANCE for f in found) for want in case.get("numbers", []))
    words = [w.lower() for w in case.get("words", [])]
    if not words:
        words_ok = True
    elif case.get("any_word"):
        words_ok = any(w in text for w in words)
    else:
        words_ok = all(w in text for w in words)
    called = any(s["tool"] == case["must_call"] for s in trace)
    return numbers_ok and words_ok and called


def main():
    client = anthropic.Anthropic() if PROVIDER == "anthropic" else make_client()
    passed = 0
    for case in CASES:
        answer, trace = run_agent(case["q"], client)
        time.sleep(3)  # stay under free tier rate limits
        ok = check(case, answer, trace)
        passed += ok
        print(("PASS" if ok else "FAIL"), "-", case["q"])
        if not ok:
            print("   answer:", answer)
            print("   tools :", [s["tool"] for s in trace])
    print(f"\n{passed}/{len(CASES)} passed")


if __name__ == "__main__":
    main()
