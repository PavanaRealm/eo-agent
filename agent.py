"""A small tool-using LLM agent for Earth observation questions.

The model decides which tools to call. This file runs the loop:
model -> tool calls -> tool results -> model ... until a final text answer.
"""
import json
import os

from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

SYSTEM = (
    "You answer questions about satellite observations of land regions. "
    "Always use the tools to get data and to compute NDVI; never compute or guess numbers yourself. "
    "If a region or date does not exist in the data, say so clearly. "
    "Answer briefly and state the NDVI values you used."
)


def run_agent(question, client, model=None, max_steps=8):
    """Run the agent. Returns (final_answer, trace). trace lists every tool call."""
    model = model or os.environ.get("MODEL", "claude-sonnet-5-5")
    messages = [{"role": "user", "content": question}]
    trace = []
    for _ in range(max_steps):
        response = client.messages.create(
            model=model, max_tokens=1024, system=SYSTEM,
            tools=TOOL_SCHEMAS, messages=messages,
        )
        if response.stop_reason != "tool_use":
            text = "".join(b.text for b in response.content if b.type == "text")
            return text.strip(), trace
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                func = TOOL_FUNCTIONS[block.name]
                output = func(**block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": json.dumps(output)})
                trace.append({"tool": block.name, "input": block.input, "output": output})
            except Exception as e:  # report tool errors back to the model
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": f"Error: {e}", "is_error": True})
                trace.append({"tool": block.name, "input": block.input, "error": str(e)})
        messages.append({"role": "user", "content": results})
    return "Stopped: too many steps without a final answer.", trace


if __name__ == "__main__":
    import sys
    import anthropic

    question = " ".join(sys.argv[1:]) or "What is the NDVI of forest_a on 2025-07-15 and what does it mean?"
    answer, trace = run_agent(question, anthropic.Anthropic())
    for step in trace:
        print("TOOL:", step)
    print("\nANSWER:", answer)
