"""SYNTHETIC model-chosen tools example (ADR 0080): a local model may call the calculator before answering."""
import json
from pathlib import Path

from agent import samples as sm


def graph(model: str = "qwen3.5:4b"):
    nodes = [sm.N("prompt", "agent.prompt", output_field="messages", items=[
                 {"kind": "template", "role": "system", "template": "Use the calculator tool for arithmetic. Reply with the result only."},
                 {"kind": "template", "role": "user", "template": "{question}"}]),
             sm.N("solve", "agent.tool_agent", model={"provider": "ollama", "model": model, "temperature": 0.0, "seed": 7, "think": False, "max_tokens": 128, "timeout_s": 60},
                  messages_field="messages", output_field="answer", tools=["calculator"], max_tool_calls=3, calls_field="calls")]
    return sm.make_graph(nodes, sm.chain("START", "prompt", "solve", "END"),
                         {"state": [sm.S("question"), sm.S("messages", "messages"), sm.S("answer"), sm.S("calls", "list", "replace", default=[])],
                          "limits": {"maxSteps": 6, "maxModelCalls": 4, "maxToolCalls": 3, "maxSeconds": 120, "maxTokens": 8192}})


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    (root / "tool_agent_calculator.project.json").write_text(json.dumps(graph().to_json(), indent=2) + "\n")
    ui = {"schemaVersion": "1.0.0", "positions": {}, "synthetic": True,
          "description": "SYNTHETIC model-chosen tools: local qwen3.5:4b may call the calculator (effect-free) up to 3 times; every call is recorded. "
                         "qwen3.5:2b skipped the tool without thinking and answered 1234×5678 wrongly; 4b used it.",
          "defaultInput": {"question": "SYNTHETIC: what is 1234 * 5678?"}}
    (root / "tool_agent_calculator.ui.json").write_text(json.dumps(ui, indent=2) + "\n")
