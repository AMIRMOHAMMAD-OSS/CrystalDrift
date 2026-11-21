"""Validate a completed 9,046 by 20 CrystalFlow replay and hash its raw evidence."""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(folder):
    with np.load(ROOT / "revision/full/mp20/cohort.npz") as cohort:
        ids, z = cohort["ids"], cohort["z"]
    n = len(ids)
    assert n == 9046 and len(np.unique(ids)) == n
    with np.load(folder / "score-crosswalk.npz") as scores:
        assert np.array_equal(ids, scores["ids"]) and scores["scored"].all()
        rms = scores["rms"]
        assert rms.shape == (n, 20) and (rms[np.isfinite(rms)] >= 0).all()
    seen = np.zeros_like(rms, dtype=bool)
    calls = 0
    errors = Counter()
    manifest = []
    for path in sorted(folder.glob("draw*-cases*.npz")):
        score_path = folder / "scores" / path.name
        with np.load(path) as raw, np.load(score_path) as score:
            draw, first = int(raw["draw"]), int(raw["first"])
            block = slice(first, first + len(raw["ids"]))
            assert not seen[block, draw].any()
            seen[block, draw] = True
            assert np.array_equal(raw["ids"], ids[block])
            assert np.array_equal(raw["z"], z[block])
            np.testing.assert_equal(rms[block, draw], score["rms"])
            assert np.isfinite(raw["k"]).all() and np.isfinite(raw["f"]).all()
            assert ((0 <= raw["f"]) & (raw["f"] < 1)).all()
            assert (raw["f"][raw["z"] == 0] == 0).all()
            calls += int(score["public_api_checks"])
            errors.update(json.loads(str(score["exceptions"])))
            manifest.append((path.name, draw, first, len(raw["ids"]), float(raw["seconds"]), digest(path), digest(score_path)))
    assert seen.all()
    with (folder / "tile-manifest.csv").open("w", newline="") as f:
        out = csv.writer(f)
        out.writerow(("file", "draw", "first", "rows", "generation_seconds", "raw_sha256", "score_sha256"))
        out.writerows(manifest)
    result = dict(targets=n, draws=20, proposals=int(seen.sum()), tiles=len(manifest),
                  public_matcher_calls_checked=calls, matcher_exceptions=dict(errors),
                  summed_tile_sampling_seconds=sum(row[4] for row in manifest),
                  score_crosswalk_sha256=digest(folder / "score-crosswalk.npz"),
                  tile_manifest_sha256=digest(folder / "tile-manifest.csv"))
    (folder / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
