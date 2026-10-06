# ADR0052: sealed (encrypted, authenticated) backups

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded separately in HANDOFF§49.

Offline backups (ADR0027) are plain directories. Their SHA-256 checksums detect
accidental corruption but not deliberate changes, and anyone who can read the
backup reads every dataset path, conversation, trace and model artifact in it.
Backups are what users move to other disks or machines, so they need
confidentiality and authenticity.

Decision: two backup subcommands, `seal` and `unseal`, that wrap an already
verified backup without changing the backup format, plus `keygen`.

- `seal` verifies the backup (optionally against a recorded manifest SHA-256),
  writes it as a tar stream of regular files and directories, and encrypts that
  stream with AES-256-GCM in 1 MiB chunks. The nonce is a random 4-byte prefix
  plus the 8-byte chunk index. Each chunk's associated data is the full header,
  the index and a final-chunk flag, so header edits, flipped bits, truncation,
  reordering, splicing and trailing bytes fail authentication or framing. The
  header records the manifest SHA-256, KDF parameters and backup format. The file
  is written privately and published by hard link, never replacing a file.
- Keys: `keygen` writes 32 random bytes (base64, mode 0600, never overwrites).
  Alternatively a passphrase file of at least 12 bytes is stretched with scrypt
  (N=2^15, r=8, p=1, 16-byte random salt). Secrets are read only from files that
  are not group/other readable; they are never command-line arguments.
- `unseal` authenticates every chunk into a private temporary file before
  extracting anything, admits only canonical relative regular files and
  directories (with the tar `data` filter), checks the extracted manifest against
  the authenticated header and an optional recorded SHA-256, runs the normal
  backup verification and publishes into a new directory. Nothing is left behind
  on failure. `restore` then works unchanged.

With the key, a successful unseal means the bytes are exactly what was sealed, so
it adds authenticity that checksums alone lack. Not provided: encryption of the
live workbench, key management/escrow/rotation, recovery from a lost key or
passphrase, multiple recipients, or protection against someone who holds the key.
The `cryptography` package was already installed through pinned dependencies; it
is now declared in requirements at the same version.

Verification: 6 pytest cases (key-file and passphrase round trips including a
multi-chunk payload and full restore; wrong key/passphrase/key type; six tamper
variants; key-file permission, length and exclusivity; destination and recorded
manifest checks; CLI). Integrated `tools/recovery_smoke.py --trackers
--cache-retention --gc --sealed` seals the real seeded backup, deletes the plain
backup and the source, unseals, restores and passes every restored journey.
