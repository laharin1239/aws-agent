"""Local AWS cost agent: a small local LLM (via Ollama) picks tools; pandas does the math.

Setup:  pip install ollama pandas numpy
Run:    python agent.py
"""
import json

import ollama

import tools

MODEL = "qwen2.5:3b"

SYSTEM_PROMPT = (
    "You are an AWS cost analysis assistant. "
    "You must NEVER calculate, estimate or guess numbers yourself. "
    "For any question about costs, call one of the provided tools and base your "
    "answer only on the tool result. Keep answers short and mention the dollar amounts "
    "exactly as returned by the tool. "
    f"The data covers {tools.DF.date.min():%Y-%m-%d} to {tools.DF.date.max():%Y-%m-%d}. "
    "Only pass start_date or end_date if the user explicitly asks about specific dates; "
    "otherwise leave them out. Never invent arguments. "
    "If a tool returns an error, retry once with corrected arguments."
)

# Map tool names the model may request to the real Python functions
TOOL_FUNCTIONS = {
    "total_cost": tools.total_cost,
    "top_services": tools.top_services,
    "cost_trend": tools.cost_trend,
    "detect_anomalies": tools.detect_anomalies,
}


def run_tool(name: str, arguments: dict):
    """Run the requested tool. Errors are returned to the model instead of crashing."""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return func(**arguments)
    except Exception as exc:  # e.g. the model passed a bad argument
        return {"error": str(exc)}


def ask(question: str, history: list) -> str:
    history.append({"role": "user", "content": question})

    # Loop: the model may call several tools before it gives a final answer
    for _ in range(5):  # safety limit so we never loop forever
        response = ollama.chat(
            model=MODEL,
            messages=history,
            tools=list(TOOL_FUNCTIONS.values()),  # ollama reads each function's signature + docstring
        )
        message = response.message
        history.append(message)

        if not message.tool_calls:  # no tool requested, so this is the final answer
            return message.content

        for call in message.tool_calls:
            name = call.function.name
            args = dict(call.function.arguments)
            result = run_tool(name, args)
            print(f"  [tool] {name}({args}) -> done")
            history.append({
                "role": "tool",
                "tool_name": name,
                "content": json.dumps(result, default=str),
            })

    return "Sorry, I could not finish that request."


def main():
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("AWS cost agent ready. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            print("Agent:", ask(question, history), "\n")


if __name__ == "__main__":
    main()