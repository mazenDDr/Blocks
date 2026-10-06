"""Actual Chrome whole-layout auto-arrange using owned native services."""
import argparse
import sys
import editor_smoke as smoke


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--chrome')
    parser.add_argument('--timeout',type=float,default=180)
    args=parser.parse_args()
    if not 1 <= args.timeout <= 600:parser.error('timeout must be 1–600 seconds')
    return smoke.run(args,journey=smoke.EDITOR/'smoke/autoLayout.mjs')


if __name__=='__main__':sys.exit(main())
