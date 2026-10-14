"""Check every snapshot data file against the SHA-256 in its provenance record.

Each *.provenance.json in this folder records either one file ("file" and
"sha256") or several ("files": {name: {"sha256": ...}}). The script recomputes
the SHA-256 of each recorded file, prints one OK or FAIL line per file, then a
total. A data file in this folder that no provenance record lists is a FAIL
too, so nothing unrecorded sits next to the pinned data. Exit status 0 means
every file matches; 1 means at least one did not.

Run from the repository root:  pixi run python book/Chapter2-DataManipulation/snapshots/check_snapshots.py
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOT_DATA = {".py", ".md"}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def recorded_hashes(folder=HERE):
    """Return {file name: (expected sha256, provenance file name)} from every provenance record."""
    out = {}
    for prov_path in sorted(folder.glob("*.provenance.json")):
        prov = json.loads(prov_path.read_text())
        if "file" in prov and "sha256" in prov:
            out[prov["file"]] = (prov["sha256"], prov_path.name)
        for name, rec in prov.get("files", {}).items():
            out[name] = (rec["sha256"], prov_path.name)
    return out


def main(folder=HERE):
    expected = recorded_hashes(folder)
    n_fail = 0
    width = max(len(n) for n in expected) if expected else 20
    for name, (sha, prov_name) in sorted(expected.items()):
        path = folder / name
        if not path.is_file():
            status, note = "FAIL", f"missing (listed in {prov_name})"
        else:
            got = sha256(path)
            status = "OK" if got == sha else "FAIL"
            note = f"sha256 {got[:12]}..." if status == "OK" else f"sha256 {got[:12]}... expected {sha[:12]}... ({prov_name})"
        n_fail += status == "FAIL"
        print(f"{status:<4}  {name:<{width}}  {note}")
    unrecorded = sorted(p.name for p in folder.iterdir()
                        if p.is_file() and p.suffix not in NOT_DATA and not p.name.endswith(".provenance.json")
                        and not p.name.startswith(".") and p.name not in expected)
    for name in unrecorded:
        n_fail += 1
        print(f"FAIL  {name:<{width}}  no provenance record lists this file")
    n_total = len(expected) + len(unrecorded)
    print(f"{n_total - n_fail} of {n_total} data files OK" + ("" if n_fail == 0 else f", {n_fail} FAIL"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
