"""The same tool-using agent, but with Google's Gemini API (free tier available).

The model decides which tools to call. This file runs the loop myself
(automatic function calling is switched off): model -> tool calls -> tool results -> model
... until a final text answer.
"""
import os
import time

from google import genai
from google.genai import types

from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

SYSTEM = (
    "You answer questions about satellite observations of land regions. "
    "Always use the tools to get data and to compute NDVI; never compute or guess numbers yourself. "
    "If a region or date does not exist in the data, say so clearly. "
    "Answer briefly and state the NDVI values you used."
)

DEFAULT_MODEL = "gemini-flash-latest"  # override with the MODEL environment variable


def _config():
    declarations = [
        types.FunctionDeclaration(
            name=t["name"], description=t["description"],
            parameters_json_schema=t["input_schema"])
        for t in TOOL_SCHEMAS
    ]
    kwargs = {}
    level = os.environ.get("THINKING_LEVEL")  # optional: minimal, low, medium or high
    if level:  # less thinking = faster answers
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=level.upper())
    return types.GenerateContentConfig(
        system_instruction=SYSTEM,
        tools=[types.Tool(function_declarations=declarations)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        **kwargs,
    )


def make_client(timeout_seconds=120):
    """A Gemini client that gives up after timeout_seconds instead of waiting forever."""
    return genai.Client(http_options=types.HttpOptions(timeout=timeout_seconds * 1000))


def _generate(client, model, contents, config, retries=3):
    """Call the model. Retry on rate limits (429), busy server (503) and server timeout (504)."""
    for attempt in range(retries + 1):
        try:
            print(f"[calling {model} ...]", flush=True)
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as e:
            if getattr(e, "code", None) in (429, 503, 504) and attempt < retries:
                wait = 20 * (attempt + 1)
                print(f"[rate limited or busy (code {e.code}), waiting {wait} s]", flush=True)
                time.sleep(wait)
                continue
            raise


def run_agent(question, client, model=None, max_steps=8):
    """Run the agent. Returns (final_answer, trace). trace lists every tool call."""
    model = model or os.environ.get("MODEL", DEFAULT_MODEL)
    config = _config()
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]
    trace = []
    for _ in range(max_steps):
        response = _generate(client, model, contents, config)
        calls = response.function_calls or []
        if not calls:
            return (response.text or "").strip(), trace
        contents.append(response.candidates[0].content)  # the model's tool-call turn
        parts = []
        for call in calls:
            args = dict(call.args or {})
            try:
                output = TOOL_FUNCTIONS[call.name](**args)
                parts.append(types.Part.from_function_response(
                    name=call.name, response={"result": output}))
                trace.append({"tool": call.name, "input": args, "output": output})
            except Exception as e:  # report tool errors back to the model
                parts.append(types.Part.from_function_response(
                    name=call.name, response={"error": str(e)}))
                trace.append({"tool": call.name, "input": args, "error": str(e)})
        contents.append(types.Content(role="user", parts=parts))
    return "Stopped: too many steps without a final answer.", trace


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What is the NDVI of forest_a on 2025-07-15 and what does it mean?"
    answer, trace = run_agent(question, make_client())  # reads GEMINI_API_KEY or GOOGLE_API_KEY
    for step in trace:
        print("TOOL:", step)
    print("\nANSWER:", answer)