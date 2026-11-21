"""Measure the learned deformation of paired sampled cells on the frozen full test pools."""
import json
import os
import sys
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from data import unpack_gram


def run():
    csp = HERE.parent / "csp"
    runs = {}
    for arm in ("angular", "radial"):
        for seed in (11, 23, 37):
            prior = csp / "test" / f"prior-seed{seed}"
            drawn = csp / "test" / f"{arm}-seed{seed}"
            with np.load(prior / "crosswalk.npz") as p, np.load(drawn / "crosswalk.npz") as d:
                np.testing.assert_array_equal(p["ids"], d["ids"])
                np.testing.assert_array_equal(p["z"], d["z"])
            ks = np.load(prior / "k.npy", mmap_mode="r")
            kd = np.load(drawn / "k.npy", mmap_mode="r")
            assert ks.shape == kd.shape == (9046, 20, 6)
            volume_ratios, anisotropies = [], []
            for offset in range(0, len(ks), 100):
                a = ks[offset:offset+100].reshape(-1, 6).astype(float)
                b = kd[offset:offset+100].reshape(-1, 6).astype(float)
                # Generalized Gram eigenvalues are squared stretches of the physical update.
                eig, q = np.linalg.eigh(unpack_gram(a))
                invroot = (q / np.sqrt(eig)[:, None, :]) @ q.transpose(0, 2, 1)
                stretches = np.sqrt(np.linalg.eigvalsh(invroot @ unpack_gram(b) @ invroot))
                anisotropies.append(stretches[:, -1] / stretches[:, 0])
                volume_ratios.append(np.exp((b[:, :3] - a[:, :3]).sum(axis=1)))
            volume_ratios = np.concatenate(volume_ratios)
            anisotropies = np.concatenate(anisotropies)
            record = dict(draws=len(volume_ratios),
                          output_to_seed_volume_q05_q50_q95=np.quantile(volume_ratios, [.05,.5,.95]).tolist(),
                          largest_to_smallest_stretch_q05_q50_q95=np.quantile(anisotropies, [.05,.5,.95]).tolist(),
                          fraction_with_largest_to_smallest_stretch_below_1p01=float(np.mean(anisotropies < 1.01)))
            runs[f"{arm}-{seed}"] = record
            print(arm, seed, record, flush=True)
    output = dict(description="All 20 draws on all 9046 MP-20 test IDs, no new model fitting or target geometry input",
                  pairing="predict.py shares each seed, ID order and prior noise across methods", runs=runs)
    (HERE / "lattice_response.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
