"""Regenerate our authored YAML sources and check after each category."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CATEGORIES = "numbers money dates-times identifiers web network email-accounts phone-numbers postal-official passwords files-paths versions code-syntax dev-tooling logs text unicode structured-data multiline exclusions measurements".split()
for name in CATEGORIES:
    subprocess.run([sys.executable, str(ROOT / (name + ".py"))], check=True)
    result = subprocess.run([sys.executable, str(ROOT / "check.py")], text=True, capture_output=True)
    print(name, "PASS" if result.returncode == 0 else "FAIL", flush=True)
    if result.returncode:
        print(result.stdout)
        raise SystemExit(result.returncode)
