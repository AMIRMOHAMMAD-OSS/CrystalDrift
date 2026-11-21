# Raw binary recovery and proposal timing

This benchmark fixes the 128 held-out, two-to-eight-site binary MP-20 structures selected by the smallest SHA-256 hashes of their material IDs. The local fits use the 3,630-cell binary training split. The released uFlowCSP checkpoint used a wider MP-20 training set. Every one of the 128 IDs is present in the release's test list and absent from its training list. The supplied unit-cell composition includes the number of atoms of each species.

## Reproduce

Run from `crystal-drift`. The proposal environment uses Python 3.10, PyTorch 2.7.1+cpu, NumPy 1.26.4, timm 1.0.21, safetensors 0.4.2, pymatgen 2024.4.13, psutil and threadpoolctl. The pinned external source and its Git-object hashes are in `uflowcsp-results`. Use the 141-tensor EMA safetensors file validated by `uflowcsp_weights.py`. Its SHA-256 is `e1854363d9e78044d1202879e40a918ab50831cb1e08eb686cb53aeacaa85673`. The code accepts `--site` for optional isolated dependency directories. For the matcher use pymatgen 2024.4.13, SciPy 1.10.1, NumPy 1.26.4 and spglib 2.5.0. The analysis uses NumPy and Matplotlib.

```text
python revision/binary/propose.py --weights PATH/TO/ema.safetensors --modes extend episode ten-thousand
python revision/binary/match.py
python revision/binary/analyze.py
python revision/binary/verify.py
```

The proposal harness pins three CPU threads to logical cores 0, 2, 4 by default. Set `--cores` for a different machine. The matcher has separate `--cores` and `--watch` options, so it can score finished episodes while later models run. The optional `--verified-cache PATH` imports original 20-draw audit records for the four local arms after validating ordered IDs and every entry of each historical five-draw RMS matrix. Without the cache, `match.py --overwrite` recomputes all distances from per-target coordinates. `analyze.py --quality-only` rebuilds fixed-draw results separately. Existing episode and throughput outputs are reused on the same configuration. For a fresh timing run use a clean project copy with the generated `episode-*.npz`, `ten-thousand-*.npz` and `rms-*.npz` omitted.

## Fixed-draw records

- `uflow-20.npz` has material IDs, zero-padded atomic numbers, six log-stretch lattice values and all eight fractional-coordinate slots. Its first five draws are byte-identical to the earlier release replay. The additional 15 use a new fixed stream. `uflow-extension.json` records seeds and source digests.
- Per-fitted-seed local samples are in `drift-results/multi-exact` and `baseline-multi-results`; `data.retrieval_baseline` regenerates the train-only donors on the complete test list before selecting the 128 IDs. This preserves the original order of draws.
- Each `rms-headline-*.npz` has 128 target IDs and a 128-by-20 normalized RMS matrix. A NaN is a miss. Exceptions count as misses and each file records its exception count. The matcher uses primitive reduction, volume scaling, fractional length tolerance 0.3, site tolerance 0.5, 10-degree angle tolerance, and exact species on all sites. The raw primary endpoint uses the returned minimum RMS match. The adjacent sensitivity study tests maximum-site displacement and retained native volume.
- `headline-prefix.csv` and `headline-per-seed.csv` contain fixed-draw MR and mean best RMS among matched targets. The latter conditions on the matched target set at each budget, which can change as more candidates match. Three local fits receive equal weight. One released checkpoint has one evaluation stream. `headline-paired.csv` gives percentage-fraction differences and bootstrap intervals for every method pair, together with differences in conditional RMS and differences restricted to targets recovered by both arms. Paired comparisons between two local fits align the training seeds. The released stream is held fixed across the three local-seed contrasts.
- Intervals are 2.5% and 97.5% quantiles of 10,000 shared bootstrap resamples of the 128 target IDs. All candidates and fitted seeds for a sampled ID move together. The interval measures target sampling uncertainty conditional on the available fits. Files store match rates as fractions.

## Warm elapsed time

All five arms ran on the same Intel Core i7-12700H laptop with the same CPU affinity and thread settings. Other project jobs used separate logical cores and shared system memory and cooling. The 10,000-request time is one measured workload per arm. The individual batch records and 10th, 50th and 90th percentile batch latencies expose within-run variation.

The measured endpoint takes a CPU array of requested species and returns the species, a symmetric canonical cell, wrapped fractional positions and Cartesian coordinates in host memory. All neural forwards use FP32. The common physical cell and coordinate outputs use float64. Formula-to-order mappings and eligible donor pools are cached once. Each arm and batch size receives five untimed warm-up calls. The external model retains its official 100 padded tokens, null space group, Cartesian prior and one-step sampler. Its cell conversion, coordinate inversion and species-order restoration are charged to it.

- `episode-*.npz` contains the actual independently drawn 128-by-20 sequences, their IDs, species, lattice and fractional coordinates and a monotonically increasing timestamp for every completion in milliseconds from the episode start. Each target is generated serially at batch one. The cap is 20 and scoring is retrospective. Startup, disk writes, retrospective matching, ranking and relaxation are excluded.
- `rms-episode-*.npz` scores the exact timed draws. `clock-budgets.csv` counts completed candidates and cases with a retrospective hit at each time budget. `runtime.csv` reports the median and 10th/90th percentiles of inter-completion times over all within-target draws and available fits.
- `ten-thousand-*.npz` records direct wall time for 10,000 geometries per method, elapsed batch times, batch counts and the exact request ID sequence. Each run cycles through the same 128 cases at batch 12 with a final batch of four, using seed-11 local weights or the released checkpoint and a separate sampling stream. `settings.json` records device, library versions, precision and thread pools.
- The editable figure is `revision-binary-accuracy-time.svg`. The typeset vector PDF and accompanying table and text are written into `paper/`. Fixed-count and elapsed-time panels use separate reproducible random streams on identical targets. `clock-per-seed.csv` and `clock-paired.csv` expose seed-specific budget counts and paired hit-rate uncertainty. `verify.py` checks all timestamps, request sequences, source prefixes and RMS array shapes, then exports `target-crosswalk.csv` and a SHA-256 manifest.
