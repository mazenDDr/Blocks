"""Actual local Ollama JSON browser journey; never substitutes a fixture provider."""
import argparse
import sys
import editor_smoke as smoke
from agent.models import ollama_status


FIXTURE = "SYNTHETIC colour/count text; actual non-fixture local Ollama structured output"


def available():
    status=ollama_status()
    if not status['reachable'] or 'qwen3.5:2b' not in status['models']:
        raise ValueError('Real installed local Ollama qwen3.5:2b required; no model download, fixture substitution or false provider acceptance.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--chrome')
    parser.add_argument('--timeout',type=float,default=180)
    args=parser.parse_args()
    if not 1<=args.timeout<=600:parser.error('timeout must be1–600seconds')
    available()
    return smoke.run(args,journey=smoke.EDITOR/'smoke/jsonAgent.mjs',fixture=FIXTURE)


if __name__=='__main__':sys.exit(main())
