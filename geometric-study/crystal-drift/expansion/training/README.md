# Compact controls, sampler replication and conditional GPU benchmarking

The protocol and choice of validation metric are in [DESIGN.md](DESIGN.md). Fit and validate the four objective families with the supplied MP-20 training and validation archives. A single uninterrupted training job writes the weights at each requested pass. Local fits use Python 3.10.8 and PyTorch 2.7.1 CPU with one thread per worker. The archived one-pass runs used two CPU threads. Independent one-pass refits agree to maximum parameter differences below 0.00001 on the first two seeds for the original objectives. Conditions, records used for positives, start weights, learning rate, optimizer updates, and configuration hashes are recorded. Each WBM run from the previous release remains available separately.

Run from the crystal-drift directory. The default process count is six. Change it or supply an explicit logical-core list for a smaller machine.

~~~sh
python expansion/training/fit.py --workers 6
python expansion/training/validate.py
python expansion/training/check.py
python expansion/training/predict.py
python expansion/training/diagnose.py
~~~

Validation uses the fixed 128 smallest hashed validation material IDs, a fair energy score on the original four geometry observables, the training-only scales and five shared draws. Training losses from the four objectives use different units. The score compares their held-out geometry in common units. The new one-Euler and 16-Euler rows use the same velocity checkpoint. The test program pairs the per-target species order, candidate index and input noise with the original full-chemistry benchmark. Norms of the traceless log stretch use the orthonormal packing of the symmetric tensor in the deposited basis. The sum of its three diagonal entries is log cell volume.

In the existing structure-matching environment (pymatgen 2024.4.13, spglib 2.5.0, numba 0.59.1, NumPy 1.26.4), run the six-worker matcher. It reads the validated lattice-frame and site-subset exclusion implementation from the previous release. Passing frames go into the same pinned pymatgen calls. Targets with a reduced primitive multiplicity bypass the exclusion. Sparse additional public-API checks fail immediately on disagreement. The outcome files retain individual error flags and include held-out IDs with zero successes.

~~~sh
python expansion/training/match.py --workers 6 --chunk 64
python expansion/training/analysis.py --replicates 10000
python expansion/training/cases.py
~~~

On a compatible CUDA machine, use the official released uFlowCSP safetensors hash and pinned source recorded in revision/full/external. The measured machine was the RTX 3060 Laptop GPU with PyTorch 2.6.0+cu124, timm 1.0.21 and FP32. Host preprocessing, random sampling, decoding, and returning host arrays are included. The full pipeline starts from known exact unit-cell composition. Five warmup calls precede each run. Fixed-batch throughput uses 10,000 structures at batch 24. Per-target serial episodes use 20 calls with timestamps after host materialization. The resulting files contain the actual samples independently scored for the time curves. One selected compact checkpoint per objective (seed 11), one deterministic composition-indexing rule for retrieval, and one released checkpoint were used. Some other experiments ran on host CPU. The local GPU was reserved for each timed series.

~~~sh
python expansion/training/gpu.py --weights /path/to/ema.safetensors
python expansion/training/gpu_score.py
python expansion/training/figures.py
python expansion/training/paper.py
~~~

The timing script accepts an optional `--site /path/to/dependencies` and an explicit `--methods` subset. The scoring and prediction scripts accept `--watch` to consume completed upstream files as they arrive. Existing complete shards are reused by the same experiment. The underlying prior archived full-split scores and public uFlowCSP availability are documented in [the original full-split protocol](../../revision/full/README.md) and [the released-model replay](../../revision/full/external/README.md).

`test-case-metrics.csv.gz` provides a portable long-form table indexed by material ID, method, seed and candidate budget. Unmatched targets have a blank best RMS. Candidate-level values remain available in each ordered crosswalk.

The test cohort, ordered species and reference geometries are stored in `revision/full/mp20/cohort.npz`. In `test/<method>-seed<seed>/`, the first two axes of `k.npy`, `f.npy` and `outcomes.npz` identify a target and its draw. `outcomes.npz` retains the material IDs, NaN for an unaccepted RMS, maximum-site flags, log volume ratios and matcher error flags. The `geometry.npz` files additionally retain shape errors on all targets and the reference-lattice site diagnostics on the specified subset. The per-seed, aggregated, and target-bootstrap estimates are in `test-quality-seeds.csv`, `test-quality.csv` and `test-paired.csv`. The same naming convention in `gpu/` joins each measured cumulative completion time to the retrospectively scored structure. Confidence intervals treat target IDs as sampling units, always carrying along their entire ordered candidate and fitted-seed records. The elapsed-time figure traces each completed first match directly.

Run `python expansion/training/val_pairs.py` to rebuild the validation uncertainty table, then run `python expansion/training/verify.py` after the final reports. The verification output checks 36 checkpoint digests, candidate dimensions and hashes, ordered material IDs, every nested local MR and conditional RMS mean, seven serial timing streams and all seven 10,000-request workloads. It also independently compares the one-thread drift refits to the archived two-thread runs.
