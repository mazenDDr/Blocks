"""Read-only application schema inventory or explicit offline migration of existing files."""
import argparse
from contextlib import closing
import json
import sys
from pathlib import Path
from .schema import SchemaError, definitions, inspect, open_database


def main():
    parser = argparse.ArgumentParser(prog='python -m storage')
    parser.add_argument('command', choices=('inspect', 'migrate'))
    parser.add_argument('--workbench', required=True)
    parser.add_argument('--offline', action='store_true', help='attest that all writers are stopped')
    args = parser.parse_args()
    if args.command == 'migrate' and not args.offline:
        parser.error('migrate requires --offline after stopping all writers')
    try:
        before = inspect(args.workbench)  # preflight all existing databases before any migration
        if args.command == 'inspect':
            result = before
        else:
            root = Path(args.workbench).expanduser().resolve()
            for relative, (kind, schema) in definitions().items():
                if (root / relative).is_file():
                    with closing(open_database(root / relative, kind, schema)):
                        pass
            result = {'before': before, 'after': inspect(root), 'offline': True}
        print(json.dumps(result, indent=2))
        return 0
    except SchemaError as error:
        print(json.dumps({'error': error.code, 'message': str(error)}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
