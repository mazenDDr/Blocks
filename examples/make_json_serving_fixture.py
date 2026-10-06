"""Labelled SYNTHETIC text extraction workflow; real installed local Ollama only."""
import json
from pathlib import Path

from agent import samples as sm


def graph():
    return sm.make_graph([
        sm.N("prompt", "agent.prompt", output_field="messages", items=[
            {"kind": "template", "role": "system", "template": "Extract colour and count from the supplied teaching text. Use only the declared JSON schema."},
            {"kind": "template", "role": "user", "template": "{question}"}]),
        sm.N("extract", "agent.structured_output", messages_field="messages", output_field="result",
             model={"provider": "ollama", "model": "qwen3.5:2b", "think": False, "max_tokens": 96, "timeout_s": 20, "temperature": 0, "seed": 0},
             schema_fields=[{"name": "colour", "type": "enum", "choices": ["red", "blue"]}, {"name": "count", "type": "integer"}],
             retry={"maxRetries": 0, "feedback": True}, on_failure="fail")],
        sm.chain("START", "prompt", "extract", "END"),
        {"state": [sm.S("question"), sm.S("messages", "messages"), sm.S("result", "object", properties={"colour": "text", "count": "integer"})],
         "limits": {"maxSteps": 8, "maxSeconds": 30, "maxModelCalls": 1, "maxTokens": 2048}})


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    (root / "serving_json_agent.project.json").write_text(json.dumps(graph().to_json(), indent=2) + "\n")
    ui = {"schemaVersion": "1.0.0", "positions": {}, "synthetic": True,
          "description": "SYNTHETIC declared colour/count extraction teaching text; real local Ollama, no model-quality benchmark claim.",
          "defaultInput": {"question": "SYNTHETIC teaching text: three red balls. Extract colour red and count 3."}}
    (root / "serving_json_agent.ui.json").write_text(json.dumps(ui, indent=2) + "\n")
