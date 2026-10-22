from __future__ import annotations

import pathlib
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parents[1]
cmd = [sys.executable, "-m", "unittest", "discover", "-s", str(root / "tests"), "-v"]
raise SystemExit(subprocess.call(cmd, cwd=root))
