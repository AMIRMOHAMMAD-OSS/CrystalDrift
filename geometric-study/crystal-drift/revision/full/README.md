# Reproduce the expanded recovery experiments

Run commands from the `crystal-drift/` directory. Two Python environments were used. The fit and generation environment used Python 3.10.8, PyTorch 2.7.1 CPU and NumPy 1.26.4. The structure-matching environment used Python 3.10.8, NumPy 1.26.4, SciPy 1.10.1, pymatgen 2024.4.13, spglib 2.5.0, Numba 0.59.1 and psutil. The original `scaled/` data and frozen MP-20 checkpoints must be present. The project retained all prior outputs. [Evaluation design](DESIGN.md) · [Preparation and original model commands](../../scaled/RUN.md)

## Frozen MP-20 models

`generate.py` makes one aligned pool of 20 candidates per method, seed and material in the published 9,046-row test split. Repeated stochastic noise is paired for the neural arms. `mp20/cohort.npz` contains the SHA-256-ordered IDs, native reference structures and original release row indices. Every method restores its original first five candidates on the original 128 targets from `scaled/runs/mp20`; each corresponding original `k` and `f` array is checked bitwise. Retrieval additionally stores integer MP-20 training row indices. The `ready.json` files contain source and output hashes. Generated geometries consume approximately 0.6 GB before compression.

```sh
python revision/full/generate.py
```

In the separate structure-matching environment, run the resumable matcher and archiver. `--workers 6` was used on this host, with one thread per worker. `--stage primary` scores the complete release at budgets one and five and the original 128 at budget twenty. `--stage extension` finishes budget twenty on the other 8,918. The following single command schedules both. Use the same chunk size when resuming.

```sh
python revision/full/match_full.py --stage all --chunk 64 --workers 6
python revision/full/report.py
python revision/full/seal.py --draws 20
python revision/full/blind_check.py
python revision/full/paper.py
```

`mp20/complete-crosswalk-k20.npz` records one normalized RMS value, a maximum-site recovery indicator and the signed log native-volume ratio for each raw candidate. A missing RMS value represents a matcher miss. Its dense axes are test ID, method/seed and draw in proposal order. Native-volume agreement requires that the same candidate both passes the RMS matcher and has an absolute log-volume ratio at most `log(1.10)`. Individual compressed tiles can be regenerated independently.

### Conservative prescreen

If the target’s primitive cell has the full supplied site count, a prediction can match only if its primitive count also equals that number. Any compatible lattice basis then corresponds to an integer frame of determinant ±1 on the generated native lattice. After normalizing the compared cells to unit volume, `prescreen.py` enumerates reciprocal-norm-bounded integer lattice vectors with pymatgen’s allowed lengths and angles. For each candidate frame it applies the same necessary fractional-coordinate subset test used upstream of Cartesian assignment. The reciprocal-metric formula supplies pymatgen’s per-axis tolerances, with a small outward rounding allowance. A failed necessary test eliminates a candidate. A passing test forwards it to pymatgen, as do exceptional search bounds and targets with fewer primitive sites. Reduced reference structures are reused across proposals. The best RMS mapping certifies a maximum-site hit whenever its largest displacement passes. Ambiguous RMS hits undergo a separate maximum-site search.

The matcher independently compares full public `get_rms_dist` and `fit` calls on deterministic candidate indices throughout the release. The original 128-target comparisons provide 7,680 deposited RMS entries, and `seal.py --audit PATH/strict-mp20.json` also checks all 12 previously audited maximum-site hit totals. Any disagreement in a checked public call terminates its tile. A separate exception count is retained in every tile and checked before archiving the final results.

`blind_check.py` makes a second direct comparison without importing the accelerated matcher. It chooses 512 targets by a separate salted hash, tests all twelve method–seed arms at one hash-assigned draw apiece, and checks the 6,144 public RMS, maximum-site and volume outcomes against the archived crosswalk. `report.py` averages recovery indicators across seeds for each case. For the mean best normalized RMS it averages over the recovered cases of each fit separately and weights the three fit means equally. The bootstrap repeats these operations in each paired resample.

## WBM replication

The two additional reference fits use the same CPU, batch, training data, formula grouping and optimizer configuration as the original seed-11 fit. Each consumes all 205,091 training structures as positive anchors within 12,819 updates. The paired experiment uses the existing 128 held-out test targets. The first two commands use the PyTorch environment.

```sh
python scaled/experiment.py --data scaled/wbm-data --out revision/full/wbm --seeds 23 37 --device cpu --threads 1 --epochs 1
python scaled/evaluate.py --data scaled/wbm-data --out revision/full/wbm --seeds 23 37
```

Use the matcher environment for the following commands. `wbm_integrate.py` links results by exact material ID and copies the small historical seed-11 metric records into the new summary directory. The original checkpoint and proposal archives keep their original paths. The common summary and bootstrap tools need NumPy.

```sh
python scaled/match.py --run revision/full/wbm --data scaled/wbm-data --workers 2
python revision/full/wbm_integrate.py --workers 2
python scaled/summarize.py revision/full/wbm --seeds 11 23 37
python revision/full/report.py
```

The Windows shell and its Ubuntu WSL environment had no configured Jean Zay SSH alias, Slurm executable, or active Slurm allocation variables identified in this run. Computation took place locally. The existing Jean Zay job stages the training data and fits the model on an allocated GPU. [Jean Zay submission procedure](../../scaled/RUN.md)
