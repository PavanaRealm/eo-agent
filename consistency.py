"""Run the benchmark several times and report how consistent the agent is.

A language model can answer differently every time. One run that passes 6 of 6 does not show
that the agent is reliable. This script runs every benchmark case N times and reports:
  - how often each case passed (a case that passes only sometimes is "flaky"),
  - how many cases passed in each run,
  - which tool sequences the agent used for each case (does it always take the same path?),
  - how long each call took,
  - API errors (timeouts, overload), counted separately from wrong answers.

Run with a real key:   GEMINI_API_KEY=... python consistency.py        (5 runs)
                       RUNS=10 GEMINI_API_KEY=... python consistency.py
Results are also written to consistency_results.json.
Note: N runs make about N x 6 questions, each with several model calls. Mind the free-tier limits.
"""
import json
import os
import statistics
import sys
import time
from collections import Counter

import eval as bench


def run_once(client, run_agent_fn, cases, check_fn, pause=3.0):
    """Run every case once. An API error is recorded as an error, not as a wrong answer."""
    rows = []
    for case in cases:
        start = time.time()
        try:
            answer, trace = run_agent_fn(case["q"], client)
            tools = [step["tool"] for step in trace]
            ok = bool(check_fn(case, answer, trace))
            error = None
        except Exception as exc:  # network, rate limit, timeout, malformed model output
            answer, tools, ok = "", [], False
            error = f"{type(exc).__name__}: {exc}"
        rows.append({
            "q": case["q"], "ok": ok, "error": error, "tools": tools,
            "answer": answer, "seconds": round(time.time() - start, 2),
        })
        time.sleep(pause)
    return rows


def summarize(runs, cases):
    """runs: a list of runs, each a list of rows from run_once. Returns a summary dict."""
    per_case = []
    for i, case in enumerate(cases):
        rows = [run[i] for run in runs]
        answered = [r for r in rows if r["error"] is None]
        passes = sum(r["ok"] for r in answered)
        paths = Counter(" > ".join(r["tools"]) or "(no tool)" for r in answered)
        per_case.append({
            "q": case["q"],
            "runs": len(rows),
            "errors": len(rows) - len(answered),
            "passes": passes,
            "answered": len(answered),
            # flaky = passed in some answered runs and failed in others
            "flaky": 0 < passes < len(answered),
            "tool_paths": dict(paths),
            "mean_seconds": round(statistics.mean(r["seconds"] for r in rows), 2),
        })
    per_run = [sum(r["ok"] for r in run) for run in runs]
    return {
        "n_runs": len(runs),
        "n_cases": len(cases),
        "passed_per_run": per_run,
        "all_cases_passed_in_every_run": all(p == len(cases) for p in per_run),
        "total_errors": sum(c["errors"] for c in per_case),
        "cases": per_case,
    }


def print_report(summary):
    n = summary["n_runs"]
    print(f"\nConsistency over {n} runs ({summary['n_cases']} cases each)")
    print("Passed per run:", summary["passed_per_run"])
    print(f"Runs with all cases passed: {sum(p == summary['n_cases'] for p in summary['passed_per_run'])}/{n}")
    print(f"API errors (not counted as wrong answers): {summary['total_errors']}\n")
    for c in summary["cases"]:
        flag = "  FLAKY" if c["flaky"] else ""
        print(f"{c['passes']}/{c['answered']} passed, {c['errors']} errors, "
              f"{len(c['tool_paths'])} tool path(s), {c['mean_seconds']}s avg - {c['q']}{flag}")
        if len(c["tool_paths"]) > 1:
            for path, count in c["tool_paths"].items():
                print(f"      {count}x  {path}")


def main():
    runs_wanted = int(os.environ.get("RUNS", sys.argv[1] if len(sys.argv) > 1 else 5))
    if bench.PROVIDER == "anthropic":
        client = bench.anthropic.Anthropic()
    else:
        client = bench.make_client()
    runs = []
    for k in range(runs_wanted):
        print(f"--- run {k + 1}/{runs_wanted} ---")
        runs.append(run_once(client, bench.run_agent, bench.CASES, bench.check))
        print("passed this run:", sum(r["ok"] for r in runs[-1]), "/", len(bench.CASES))
    summary = summarize(runs, bench.CASES)
    print_report(summary)
    with open("consistency_results.json", "w") as f:
        json.dump({"summary": summary, "runs": runs}, f, indent=2)
    print("\nSaved consistency_results.json")


if __name__ == "__main__":
    main()
