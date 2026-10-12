"""Execute every notebook in the book TOC sequentially, in place.

MyST's --execute runs ~10 kernels concurrently, which exhausts CI-runner
memory (torch + sklearn + obspy kernels at once) and kills the Jupyter
server. This script executes one notebook at a time with nbconvert, each from
its own directory, so `myst build --html` can then render the stored outputs.

With --cache-dir, a notebook is executed only if its fingerprint changed since
the cached run. The fingerprint covers what can change its outputs: the code
cells (source, in order), the kernel name, the locked environment (pixi.lock),
the course package (book/mlgeo_synth), and data-like files stored beside the
notebook (for example 2.6's snapshots/). On a cache hit the cached outputs are
copied into the current notebook, so markdown edits never trigger a rerun.
Notebooks that download living data (NGL, USGS, FDSN) keep their cached
outputs until a full run: CI runs one weekly with --no-cache.

Usage: python tools/execute_notebooks.py [--only substring] [--cache-dir DIR] [--no-cache]
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BOOK = ROOT / "book"
DATA_SUFFIXES = {".csv", ".json", ".txt", ".py", ".nc", ".parquet", ".h5", ".hdf5", ".npz", ".npy",
                 ".yaml", ".yml", ".mseed", ".sac", ".gz", ".zip", ".dat", ".tsv", ".xml"}
SKIP_DIRS = {"data", "_build", ".ipynb_checkpoints", "__pycache__"}


def toc_notebooks():
    cfg = yaml.safe_load((BOOK / "myst.yml").read_text())

    def walk(entries):
        for e in entries:
            f = e.get("file", "")
            if f.endswith(".ipynb"):
                yield BOOK / f
            yield from walk(e.get("children", []))

    yield from walk(cfg["project"]["toc"])


def file_digest(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def global_digest():
    """Inputs shared by every notebook: the locked environment and the course package."""
    paths = [ROOT / "pixi.lock"] + sorted((BOOK / "mlgeo_synth").rglob("*.py"))
    return file_digest([p for p in paths if p.is_file()])


_dir_cache = {}


def local_data_digest(folder):
    """Data-like files stored beside a notebook (not downloads: data/ is git-ignored)."""
    if folder not in _dir_cache:
        files = [p for p in folder.rglob("*")
                 if p.is_file() and p.suffix in DATA_SUFFIXES
                 and not SKIP_DIRS.intersection(p.relative_to(folder).parts)]
        _dir_cache[folder] = file_digest(files)
    return _dir_cache[folder]


def fingerprint(nb_path, shared):
    nb = json.loads(nb_path.read_text())
    h = hashlib.sha256()
    h.update(shared.encode())
    h.update(local_data_digest(nb_path.parent).encode())
    h.update(nb.get("metadata", {}).get("kernelspec", {}).get("name", "").encode())
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            h.update(b"\x00" + "".join(c["source"]).encode())
    return h.hexdigest()


def restore_outputs(nb_path, cached_path):
    """Copy outputs from the cached executed notebook into the current one (code cells match)."""
    cur = json.loads(nb_path.read_text())
    old = json.loads(cached_path.read_text())
    cur_code = [c for c in cur["cells"] if c["cell_type"] == "code"]
    old_code = [c for c in old["cells"] if c["cell_type"] == "code"]
    if len(cur_code) != len(old_code) or any(
            "".join(a["source"]) != "".join(b["source"]) for a, b in zip(cur_code, old_code)):
        return False
    for a, b in zip(cur_code, old_code):
        a["outputs"] = b.get("outputs", [])
        a["execution_count"] = b.get("execution_count")
    if "language_info" in old.get("metadata", {}):
        cur.setdefault("metadata", {})["language_info"] = old["metadata"]["language_info"]
    nb_path.write_text(json.dumps(cur, indent=1, ensure_ascii=False) + "\n")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="only notebooks whose path contains this")
    ap.add_argument("--timeout", type=int, default=1200, help="seconds per notebook")
    ap.add_argument("--cache-dir", default="", help="reuse outputs of unchanged notebooks from this folder")
    ap.add_argument("--no-cache", action="store_true", help="execute everything, then refresh the cache")
    args = ap.parse_args()

    cache = Path(args.cache_dir).resolve() if args.cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    shared = global_digest() if cache else ""

    failures, n_run, n_hit, t_start = [], 0, 0, time.time()
    for nb in toc_notebooks():
        if args.only and args.only not in str(nb):
            continue
        rel = nb.relative_to(ROOT)
        key = fingerprint(nb, shared) if cache else ""
        cached = cache / f"{key}.ipynb" if cache else None
        if cache and not args.no_cache and cached.exists() and restore_outputs(nb, cached):
            n_hit += 1
            print(f"{'hit':>4}  {'':>6}   {rel}", flush=True)
            continue

        t0 = time.time()
        r = subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
             "--execute", "--inplace", f"--ExecutePreprocessor.timeout={args.timeout}",
             nb.name],
            cwd=nb.parent, capture_output=True, text=True,
        )
        dt = time.time() - t0
        n_run += 1
        status = "ok" if r.returncode == 0 else "FAIL"
        print(f"{status:>4}  {dt:6.1f}s  {rel}", flush=True)
        if r.returncode != 0:
            print(r.stderr[-3000:], flush=True)
            failures.append(str(rel))
        elif cache:
            shutil.copyfile(nb, cached)

    if cache:
        print(f"\nexecuted {n_run}, reused {n_hit} from cache, {time.time() - t_start:.0f}s total")
    if failures:
        print(f"\n{len(failures)} notebook(s) failed:", *failures, sep="\n  ")
        return 1
    print("\nall notebooks executed clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
