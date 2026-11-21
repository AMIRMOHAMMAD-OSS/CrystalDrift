"""Post hoc component diagnostic on the frozen 128-ID, five-draw validation cohort."""
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "revision/full"))
from match_full import candidate, target_context, unpack_gram


def run():
    assert version("pymatgen") == "2024.4.13"
    with np.load(ROOT / "scaled/mp20-data/val.npz") as f:
        val = {name: f[name] for name in ("ids", "z", "k", "f")}
    val_index = {str(x): i for i, x in enumerate(val["ids"])}
    arms = ("as_generated", "true_volume_only", "true_lattice", "true_fractional_sites", "true_both")
    runs = {}
    for seed in (11, 23, 37):
        fit = HERE.parent / "csp/fits" / f"angular-{seed}" / "val-00256.npz"
        with np.load(fit) as f:
            ids, z, cells, sites = (f[name] for name in ("ids", "z", "k", "f"))
        cases = len(ids)
        cells = cells.reshape(cases, 5, 6)
        sites = sites.reshape(cases, 5, 20, 3)
        hits = np.zeros((len(arms), cases, 5), dtype=bool)
        native = np.zeros_like(hits)
        for i, (mid, chem) in enumerate(zip(ids, z)):
            target_ix = val_index[str(mid)]
            np.testing.assert_array_equal(chem, val["z"][target_ix])
            kt, ft = val["k"][target_ix], val["f"][target_ix]
            ctx = target_context(kt, ft, chem)
            active = chem[chem > 0].astype("int64")
            vg = cells[i, :, :3].sum(axis=1, dtype="float64")
            vt = kt[:3].sum(dtype="float64")
            for j, (kg, fg) in enumerate(zip(cells[i], sites[i])):
                volume_only = kg.astype("float64")
                volume_only[:3] += (vt - vg[j]) / 3.
                for ai, (k, f) in enumerate(((kg, fg), (volume_only, fg), (kt, fg), (kg, ft), (kt, ft))):
                    if ai == 4 and j > 0:
                        hits[ai, i, j] = hits[ai, i, 0]
                        native[ai, i, j] = native[ai, i, 0]
                        continue
                    rms, _, _ = candidate(ctx, k, f, active, unpack_gram(k), False)
                    hits[ai, i, j] = np.isfinite(rms)
                    native[ai, i, j] = np.isfinite(rms) and abs(k[:3].sum(dtype="float64") - vt) <= np.log(1.1)
        assert hits[-1].all() and native[-1].all()
        result = {arm: dict(scaled_hits_1=int(h[:,0].sum()), scaled_hits_5=int(h.any(1).sum()),
                            native_hits_1=int(n[:,0].sum()), native_hits_5=int(n.any(1).sum()))
                  for arm, h, n in zip(arms, hits, native)}
        runs[seed] = result
        print(seed, json.dumps(result), flush=True)
    out = dict(cohort="Frozen original validation panel", targets=cases, draws=5, seeds=list(runs),
               coordinate_convention="native deposited fractional chart", native="matched and absolute log volume ratio <= log(1.1)",
               method="oracle component injection, no generator retraining", runs=runs)
    (HERE / "oracle_validation.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
