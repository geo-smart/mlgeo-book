"""Check that every course data file in this folder is exactly the file recorded at download time.

Each *.provenance.json in this folder records, for one file ("file" and "sha256") or for several
("files": {name: {"sha256": ..., "bytes": ...}}), the file's fingerprint and usually its size in
bytes. The fingerprint is the SHA-256 hash of the file: a 64-character code computed from every
byte, so changing a single byte gives a completely different code. If the code computed now equals
the recorded one, the file on your disk is byte for byte the course copy; if not, it was edited,
cut short or damaged.

The script prints one line per file (OK or FAIL, its size, the start of its fingerprint), then one
sentence in plain words. A data file in this folder that no provenance record lists is a FAIL too,
so nothing without a recorded source sits next to the course copies. Exit status 0 means every
file matches; 1 means at least one did not.

Run from the repository root:
    pixi run python book/Chapter2-DataManipulation/snapshots/check_snapshots.py
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOT_DATA = {".py", ".md"}


def sha256(path, chunk=1 << 20):
    """The file's fingerprint: SHA-256 of its bytes, as 64 hexadecimal characters."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def recorded_hashes(folder=HERE):
    """Return {file name: (recorded fingerprint, provenance file name)} from every record."""
    out = {}
    for prov_path in sorted(folder.glob("*.provenance.json")):
        prov = json.loads(prov_path.read_text())
        if "file" in prov and "sha256" in prov:
            out[prov["file"]] = (prov["sha256"], prov_path.name)
        for name, rec in prov.get("files", {}).items():
            out[name] = (rec["sha256"], prov_path.name)
    return out


def recorded_sizes(folder=HERE):
    """Return {file name: recorded size in bytes} for the files whose record states a size."""
    out = {}
    for prov_path in sorted(folder.glob("*.provenance.json")):
        prov = json.loads(prov_path.read_text())
        if "file" in prov and "sha256" in prov and isinstance(prov.get("bytes"), int):
            out[prov["file"]] = prov["bytes"]
        for name, rec in prov.get("files", {}).items():
            if isinstance(rec.get("bytes"), int):
                out[name] = rec["bytes"]
    return out


def main(folder=HERE):
    expected = recorded_hashes(folder)
    sizes = recorded_sizes(folder)
    print(f"Course data files in {folder.name}/, compared with the size and fingerprint "
          "(SHA-256) recorded when each was downloaded.")
    print("Change one byte of a file and its fingerprint changes completely, "
          "so a match means the file is exactly the recorded one.\n")
    n_fail = 0
    width = max(len(n) for n in expected) if expected else 20
    for name, (sha, prov_name) in sorted(expected.items()):
        path = folder / name
        if not path.is_file():
            status, note = "FAIL", f"missing: {prov_name} lists it, but it is not on disk"
        else:
            got, size = sha256(path), path.stat().st_size
            status = "OK" if got == sha else "FAIL"
            if status == "OK":
                note = f"{size:>11,} bytes  fingerprint {got[:12]}... matches"
            else:
                was = sizes.get(name)
                size_note = "" if was in (None, size) else f" (recorded size {was:,} bytes)"
                note = (f"{size:>11,} bytes{size_note}  fingerprint {got[:12]}... differs from "
                        f"the recorded {sha[:12]}... in {prov_name}: the file was changed or damaged")
        n_fail += status == "FAIL"
        print(f"{status:<4}  {name:<{width}}  {note}")
    unrecorded = sorted(p.name for p in folder.iterdir()
                        if p.is_file() and p.suffix not in NOT_DATA
                        and not p.name.endswith(".provenance.json")
                        and not p.name.startswith(".") and p.name not in expected)
    for name in unrecorded:
        n_fail += 1
        print(f"FAIL  {name:<{width}}  no provenance record lists this file, "
              "so its source and fingerprint are unknown")
    n_total = len(expected) + len(unrecorded)
    print()
    if n_fail == 0:
        print(f"All {n_total} data files match their recorded fingerprints: "
              "none was changed or damaged.")
    else:
        print(f"{n_fail} of {n_total} data files do not match their record (lines marked FAIL). "
              "`git status` shows which files changed; "
              "`git checkout -- book/Chapter2-DataManipulation/snapshots/` "
              "restores the course copies.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
