"""Owned native pinned-retrieval / short-term policy conversation browser journey."""
import argparse
import sys
import editor_smoke as smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--chrome')
    parser.add_argument('--json', action='store_true', help='Use actual installed local Ollama structured output; no substitute provider.')
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 600:
        parser.error('timeout must be 1–600 seconds')
    if args.json:
        from editor_json_agent_smoke import available
        available()
    return smoke.run(args, journey=smoke.EDITOR / 'smoke/contextAgent.mjs',
                     extra_env={'VOID_CONTEXT_JSON': '1' if args.json else '0'},
                     fixture='SYNTHETIC pinned retrieval + bounded native short-term conversation' + ('; actual Ollama JSON' if args.json else '; zero model calls'))


if __name__ == '__main__':
    sys.exit(main())
