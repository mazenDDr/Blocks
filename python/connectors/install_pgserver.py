"""Install `pgserver` (embedded real PostgreSQL binaries) into the current environment.

    python -m connectors.install_pgserver

pgserver 0.1.4 publishes wheels for CPython 3.9-3.12 only, but the wheel is pure Python plus PostgreSQL 16 binaries (nothing links against the
interpreter). On a newer Python this downloads the CPython 3.12 wheel for this platform and installs it re-tagged `py3-none-<platform>`.
On Python <= 3.12 a plain `pip install pgserver==0.1.4` works and this module is not needed."""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

VERSION = "0.1.4"


def platform_tag() -> str:
    m, s = platform.machine().lower(), platform.system()
    if s == "Darwin":
        return "macosx_11_0_arm64" if m in ("arm64", "aarch64") else "macosx_10_9_x86_64"
    if s == "Linux" and m in ("x86_64", "amd64"):
        return "manylinux_2_17_x86_64.manylinux2014_x86_64"
    raise SystemExit(f"pgserver has no wheel for {s} {m}")


def main() -> None:
    tag = platform_tag()
    pip = [sys.executable, "-m", "pip"]
    subprocess.run([*pip, "install", "fasteners>=0.19", "platformdirs>=4.0.0", "psutil>=5.9.0"], check=True)
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([*pip, "download", f"pgserver=={VERSION}", "--no-deps", "--only-binary=:all:", "--python-version", "3.12", "--platform", tag, "-d", d], check=True)
        whl = next(Path(d).glob("pgserver-*.whl"))
        retagged = whl.with_name(whl.name.replace("cp312-cp312", "py3-none"))
        shutil.move(whl, retagged)
        subprocess.run([*pip, "install", "--no-deps", "--force-reinstall", str(retagged)], check=True)


if __name__ == "__main__":
    main()
