"""SYNTHETIC agent evaluation example (ADR 0077): a local-model question answerer and 20 cases with computable answers."""
import json
import random
from pathlib import Path

from agent import samples as sm


def graph(model: str = "qwen3.5:2b"):
    nodes = [sm.N("prompt", "agent.prompt", output_field="messages", items=[
                 {"kind": "template", "role": "system", "template": "Answer with only the requested value, no explanation."},
                 {"kind": "template", "role": "user", "template": "{question}"}]),
             sm.N("reply", "agent.chat_model", messages_field="messages", output_field="answer",
                  model={"provider": "ollama", "model": model, "temperature": 0.0, "max_tokens": 32, "seed": 7, "think": False, "timeout_s": 60})]
    return sm.make_graph(nodes, sm.chain("START", "prompt", "reply", "END"),
                         {"state": [sm.S("question"), sm.S("messages", "messages"), sm.S("answer")], "limits": {"maxSteps": 6, "maxModelCalls": 1, "maxSeconds": 60}})


def cases():
    rng = random.Random(20261006)
    out = []
    for i in range(12):  # two-operand arithmetic with exact numeric answers
        a, b = rng.randint(12, 99), rng.randint(12, 99)
        op = ["+", "-", "*"][i % 3]
        value = a + b if op == "+" else a - b if op == "-" else a * b
        out.append({"id": f"arith-{i:02d}", "input": {"question": f"SYNTHETIC: what is {a} {op} {b}? Reply with the number only."},
                    "checks": [{"field": "answer", "kind": "number_close", "value": value, "tolerance": 0}]})
    words = ["harbor", "lantern", "meadow", "quartz", "violet", "cobalt", "orchid", "summit"]
    for i, w in enumerate(words[:4]):  # deterministic string transforms
        out.append({"id": f"upper-{i}", "input": {"question": f"SYNTHETIC: write the word {w} in capital letters. Reply with the word only."},
                    "checks": [{"field": "answer", "kind": "equals", "value": w.upper(), "case_sensitive": True}]})
    for i, w in enumerate(words[4:]):
        out.append({"id": f"reverse-{i}", "input": {"question": f"SYNTHETIC: spell the word {w} backwards. Reply with the reversed word only."},
                    "checks": [{"field": "answer", "kind": "equals", "value": w[::-1]}]})
    return out


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    (root / "evaluation_arithmetic.project.json").write_text(json.dumps(graph().to_json(), indent=2) + "\n")
    ui = {"schemaVersion": "1.0.0", "positions": {}, "synthetic": True,
          "description": "SYNTHETIC evaluation example: a local Ollama model answers 20 questions with computable answers (arithmetic, capitalisation, "
                         "reversal); the Evaluate tab scores them with literal checks. A pass rate over these cases, not a general quality claim.",
          "defaultInput": {"question": "SYNTHETIC: what is 12 + 30? Reply with the number only."}, "evaluationCases": cases()}
    (root / "evaluation_arithmetic.ui.json").write_text(json.dumps(ui, indent=2) + "\n")
