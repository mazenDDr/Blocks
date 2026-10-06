"""python -m workbench_backup {create,verify,restore,keygen,seal,unseal} --help"""
import argparse
import json
import sys

from .core import BackupError, create, restore, verify
from .sealed import generate_key, seal, unseal


def main():
    parser = argparse.ArgumentParser(description="Offline trusted local workbench backup/recovery (ADR0027).")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("create", help="Stop ALL source writers first; snapshot into a new directory.")
    c.add_argument("source")
    c.add_argument("destination")
    c.add_argument("--offline", action="store_true", help="Attest that all control/worker/tracker/repository writers are stopped.")
    c.add_argument("--links", choices=("reject", "internal"), default="reject", help="Opt in to v2 relative internal link preservation; backups store link metadata without traversal.")
    c.add_argument("--omit-wandb-external-logs", action="store_true", help="With --links internal, explicitly omit native external debug-core.log links and record their targets.")
    v = sub.add_parser("verify", help="Check inventory, hashes, SQLite, CAS and direct database references.")
    v.add_argument("backup")
    v.add_argument("--manifest-sha256")
    r = sub.add_parser("restore", help="Recover into a NEW workbench; absolute paths/identities remain unchanged.")
    r.add_argument("backup")
    r.add_argument("destination")
    r.add_argument("--trusted-local", action="store_true", help="Attest trust in backup provenance; checksums are not authentication.")
    r.add_argument("--manifest-sha256")
    k = sub.add_parser("keygen", help="Write a new random 256-bit key file (mode 0600); never overwrites.")
    k.add_argument("key_file")
    for name, text in (("seal", "Encrypt and authenticate a verified backup into one new file."),
                       ("unseal", "Authenticate a sealed file, then extract and verify it into a NEW backup directory.")):
        s = sub.add_parser(name, help=text)
        s.add_argument("source")
        s.add_argument("destination")
        secret = s.add_mutually_exclusive_group()
        secret.add_argument("--key-file", help="base64 32-byte key file from keygen (mode 0600)")
        secret.add_argument("--passphrase-file", help="file holding a passphrase of at least 12 bytes (mode 0600); scrypt-derived")
        s.add_argument("--manifest-sha256")
    args = parser.parse_args()
    try:
        if args.command == "create":
            result = create(args.source, args.destination, offline=args.offline, links=args.links,
                            omit_wandb_external_logs=args.omit_wandb_external_logs)
        elif args.command == "keygen":
            result = generate_key(args.key_file)
        elif args.command in ("seal", "unseal"):
            result = (seal if args.command == "seal" else unseal)(args.source, args.destination, key_file=args.key_file,
                                                               passphrase_file=args.passphrase_file, manifest_sha256=args.manifest_sha256)
        elif args.command == "restore":
            result = restore(args.backup, args.destination, trusted=args.trusted_local, manifest_sha256=args.manifest_sha256)
        else:
            m = verify(args.backup, manifest_sha256=args.manifest_sha256)
            result = {"verified": True, "files": len(m["files"]), "databases": m["databases"], "policy": m["policy"],
                      "links": len(m.get("links", {})), "omittedLinks": m.get("omittedLinks", {})}
        print(json.dumps(result, indent=2))
        return 0
    except (BackupError, OSError) as e:
        print(json.dumps({"error": {"code": e.code if isinstance(e, BackupError) else "E_BACKUP_IO", "message": str(e)}}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
