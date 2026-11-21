"""Compare paired five- and hundred-step CrystalFlow proposals from the same initial noise."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(steps):
    folder = HERE / f"csp-s{steps}"
    opts = json.loads((folder / "settings.json").read_text())
    assert opts["steps"] == steps and opts["tf32"] and opts["batch"] == 24
    assert opts.get("coord_slope",0.) == (5. if steps==100 else 0.)
    with np.load(folder / "score-crosswalk.npz") as z:
        assert z["scored"][:128, 0].all()
        return z["ids"][:128], z["rms"][:128, 0]


if __name__ == "__main__":
    ids, low = load(5)
    ids2, high = load(100)
    np.testing.assert_equal(ids, ids2)
    left = np.isfinite(low)
    right = np.isfinite(high)
    both = left & right
    diff = right.astype(float) - left
    rng = np.random.default_rng(9091)
    bootstrap = diff[rng.integers(0, len(ids), size=(10000, len(ids)))].mean(1)
    result = dict(targets=len(ids), draws=1, tf32=True, steps=[5, 100],coord_slopes=[0.,5.],
                  hits=[int(left.sum()), int(right.sum())], both=int(both.sum()),
                  only_5=int((left & ~right).sum()), only_100=int((~left & right).sum()),
                  neither=int((~left & ~right).sum()), mr_gap=float(diff.mean()),
                  mr_gap_95_ci=np.quantile(bootstrap, [.025, .975]).tolist(),
                  mean_rms=[float(low[left].mean()), float(high[right].mean())],
                  common_hit_mean_rms=[float(low[both].mean()), float(high[both].mean())])
    (HERE / "steps-5-vs-100.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
