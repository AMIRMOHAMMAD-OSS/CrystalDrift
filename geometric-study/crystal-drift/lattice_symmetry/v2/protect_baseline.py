"""Check the immutable fixed-basis release and the prototype that started v2."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def baseline():
    original = {}
    for row in (ROOT/"release-sha256.txt").read_text().splitlines():
        value, name = row.split(maxsplit=1)
        archive = ROOT/name
        if name == "crystal-drift-research.zip" and not archive.is_file():
            archive = ROOT.parent/name
        actual = sha(archive)
        if actual != value:
            raise ValueError(f"Fixed-basis release mismatch for {name}: {actual} vs {value}")
        original[name] = actual
    tracked = [*sorted((ROOT/"scaled/runs").glob("*/*.pt")),
               *sorted((ROOT/"expansion/training").rglob("*.pt")),
               *sorted(HERE.parent.glob("*.py")), HERE.parent/"CODEX_GOAL.md"]
    return dict(fixed_release=original,
                checkpoints_and_prototype={p.relative_to(ROOT).as_posix():sha(p) for p in tracked})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--record",action="store_true")
    args = p.parse_args()
    dst = HERE/"baseline.json"
    snap = baseline()
    if not dst.exists() and args.record:
        dst.write_text(json.dumps(snap, indent=2)+"\n")
    if json.loads(dst.read_text()) != snap:
        raise ValueError("Baseline changed")
    print(f"Verified {len(snap['fixed_release'])} release files and {len(snap['checkpoints_and_prototype'])} checkpoint/prototype inputs.")
