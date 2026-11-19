# EO Agent: a small tool-using LLM agent for Earth observation questions

A Python agent that answers questions about satellite observations. The model decides which
tools to call (look up a region, compute NDVI, classify land cover). The loop in `agent.py`
runs the calls and sends the results back until the model gives a final answer.

NDVI = (NIR - Red) / (NIR + Red). The sample data in `data/regions.csv` is synthetic.

## Files
- `tools.py`: the three tools and their schemas
- `agent_gemini.py`: the agent loop with the Gemini API (free tier). Automatic function calling is off, so the loop is written out; tool errors are returned to the model; step limit; retry on rate limits
- `agent.py`: the same loop with the Anthropic API
- `eval.py`: a small benchmark of 6 questions; checks the numbers in the answer AND that the right tool was called
- `list_models.py`: lists the models your key can use
- `tests/`: offline tests of tools and both loops with a fake model

## Run (Gemini, free tier)
    pip install -r requirements.txt
    python -m pytest -q                      # offline tests, no API key needed
    export GEMINI_API_KEY=...                # key from Google AI Studio
    python agent_gemini.py "Did vegetation in crop_b decline between 2025-05-10 and 2025-09-10?"
    python eval.py                           # benchmark with the real model

If the model name fails, run `python list_models.py` and set `export MODEL=<a name from the list>`.
To use Anthropic instead: `PROVIDER=anthropic ANTHROPIC_API_KEY=... python eval.py`.

## Results
Run with `gemini-3.1-flash-lite` on the Gemini API free tier:
- First run: 4 of 6 cases passed. One failure was a bug in my checker: it looked for the exact text `0.7273`, but the agent correctly answered "approximately 0.727".
- I changed the checker to compare numbers with a tolerance of 0.005 (two-decimal rounding counts as correct) and added tests for the checker.
- After the fix: 6 of 6 passed (one run, so this is not a statistical result).

The loop retries on 429, 503 and 504, because the Gemini API sometimes returns these errors.

## Limits
- Only three tools and six synthetic observations, so this is a learning project, not a real EO system.
- No real satellite imagery and no vision-language model yet.
- The benchmark has only six questions; the consistency check used five runs of one model. It checks numbers and words, so an unusual wording can still fail.
- The agent loop is plain Python code, not an agent framework. I built this project with AI assistants (Claude, Cursor) and tested it.

## Consistency (5 runs)
`consistency.py` ran the 6-question benchmark 5 times with `gemini-3.1-flash-lite` on 6 November 2026.
- Passed per run: 6, 6, 6, 6, 6. All 30 answers passed; 0 API errors.
- Every question used the same tool path in all 5 runs.
- Average time per question: 8 to 18 seconds.
- This shows consistency for this model and these six questions on synthetic data. It does not show
  how the agent behaves with other models, harder questions or real data.
