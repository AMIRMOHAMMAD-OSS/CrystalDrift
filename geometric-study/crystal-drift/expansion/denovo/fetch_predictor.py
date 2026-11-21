"""Download verified public property evaluator artifacts at the pinned DiffCSP commit."""

import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent / "property"
COMMIT = "7121d159826efa2ba9500bf299250d96da37f146"
files = [
    "diffcsp/prop_models/mp20/" + s
    for s in ("epoch=839-step=89039.ckpt", "prop_scaler.pt", "hparams.yaml", "lattice_scaler.pt")
]
files += [
    "diffcsp/pl_modules/gnn.py",
    "diffcsp/common/data_utils.py",
    "diffcsp/pl_data/dataset.py",
    "scripts/eval_utils.py",
    "scripts/compute_metrics.py",
    "LICENSE",
]
tree = json.load(
    urlopen(
        Request(
            f"https://api.github.com/repos/jiaor17/DiffCSP/git/trees/{COMMIT}?recursive=1",
            headers={"User-Agent": "CrytalDrift-research"},
        ),
        timeout=30,
    )
)
listing = {x["path"]: x for x in tree["tree"]}
out = {}
for name in files:
    info = listing[name]
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    data = (
        p.read_bytes()
        if p.exists()
        else urlopen(
            f"https://raw.githubusercontent.com/jiaor17/DiffCSP/{COMMIT}/{name}", timeout=90
        ).read()
    )
    assert len(data) == info["size"]
    assert (
        hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == info["sha"]
    )
    out[name] = hashlib.sha256(data).hexdigest()
    p.write_bytes(data)
(ROOT / "manifest.json").write_text(json.dumps(dict(commit=COMMIT, sha256=out), indent=2) + "\n")
print("Verified", len(files), "published artifacts")
