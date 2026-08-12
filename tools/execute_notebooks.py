"""Execute every notebook in the book TOC sequentially, in place.

MyST's --execute runs ~10 kernels concurrently, which exhausts CI-runner
memory (torch + sklearn + obspy kernels at once) and kills the Jupyter
server. This script executes one notebook at a time with nbconvert, each from
its own directory, so `myst build --html` can then render the stored outputs.

Usage: python tools/execute_notebooks.py [--only substring]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BOOK = ROOT / "book"


def toc_notebooks():
    cfg = yaml.safe_load((BOOK / "myst.yml").read_text())

    def walk(entries):
        for e in entries:
            f = e.get("file", "")
            if f.endswith(".ipynb"):
                yield BOOK / f
            yield from walk(e.get("children", []))

    yield from walk(cfg["project"]["toc"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="only notebooks whose path contains this")
    ap.add_argument("--timeout", type=int, default=1200, help="seconds per notebook")
    args = ap.parse_args()

    failures = []
    for nb in toc_notebooks():
        if args.only and args.only not in str(nb):
            continue
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
             "--execute", "--inplace", f"--ExecutePreprocessor.timeout={args.timeout}",
             nb.name],
            cwd=nb.parent, capture_output=True, text=True,
        )
        dt = time.time() - t0
        status = "ok" if r.returncode == 0 else "FAIL"
        print(f"{status:>4}  {dt:6.1f}s  {nb.relative_to(ROOT)}", flush=True)
        if r.returncode != 0:
            print(r.stderr[-3000:], flush=True)
            failures.append(str(nb.relative_to(ROOT)))

    if failures:
        print(f"\n{len(failures)} notebook(s) failed:", *failures, sep="\n  ")
        return 1
    print("\nall notebooks executed clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
