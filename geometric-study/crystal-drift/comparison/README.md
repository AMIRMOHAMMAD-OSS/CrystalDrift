# Evaluating generated binary cells

The official two-site AB test split contains 189 distinct structure IDs. The binary two-to-eight-site split contains 1,229 IDs, including those 189. Environment scoring uses every test structure and the first 20 AB or five multisite draws. Structural matching uses five draws for every AB target and the 128 multisite targets with the smallest SHA-256 hashes of their material IDs.

## Comparing conditional fidelity

Run the following from the crystal-drift directory after training and prediction export. The scripts use NumPy, SciPy and Matplotlib. The matcher also requires pymatgen. One BLAS thread is sufficient.

```shell
python compare.py
python plot_results.py
python verify_comparison.py
python standard_match.py --external
python external_comparison.py --require-match
```

The energy score uses standardized log volume per atom and shortest periodic A–A, A–B and B–B distances. AB has only one distinct like-species distance. The per-target score includes the empirical ensemble self-pairs. The separate fair-energy column omits self-pairs in the diversity term. The pooled geometry plot shows the primary score and the fraction of generated candidates with a contact shorter than 0.5 Å or a numerical failure. The single-file run summaries under `cache/` are keyed by hashes of evaluator, splits and inputs. The independent geometry check uses a different cell decoder and direct enumeration of periodic images.

The `groups`, `per-case` and `per-seed` tables report means and outcomes for each method. Files containing `paired-drift-minus` subtract each listed control from pooled drifting on each reference ID. The exact-composition arm was added after viewing initial test results. The `val-*` files report its locked comparison with pooled drifting on validation. The `paired-exact-minus` validation contrasts subtract pooled drifting from exact-composition drifting. All intervals use 10,000 paired case-bootstrap samples after averaging the three fits per structure. They condition on those fits and their generated draws. The 32 and 239 test rows labeled seen match a training composition with exactly the same ordered elements and cell multiplicities.

The `external-ab/` and `external-multi/` directories cover all compact methods and the full-MP-20 pretrained uFlowCSP checkpoint at five candidates per target. They use the same 189 AB and 128 multisite structures used for structural matching. They include one published checkpoint and one sample seed for uFlowCSP, with three fitted seeds for each compact neural method. External decoding used its own hardware and batch protocol. The per-cohort replay records under `../uflowcsp-results/` report resource costs.

## Estimating recovery within a time budget

```shell
python frontier.py
python time_frontier.py
python verify_frontier.py
```

The scripts consume the loaded-checkpoint CPU and RTX timing files in `../speed-results/` and draw-level outcomes in `../structure-matching/`. They write outputs in `../frontier-results/`. Five structural matches per target are observed. The secondary environment criterion accepts a draw when all four physical descriptors fall within a ratio of 1/1.15 to 1.15 of the target. That criterion uses all 20 available AB or five multisite environment evaluations. Both are oracle criteria that require the reference geometry.

- `fixed-candidate.csv` gives empirical recovery and 95% case intervals for each fixed prefix. `first-hit-by-case.csv` lists the first successful index or an empty field when there was no event through the cap.
- `prefix-cost.csv` gives the coordinates of the speed figure. For each method and prefix of length k, estimated serial generation cost is k times the mean of the three per-seed median batch-one call times. Recovery is averaged over the observed ordered prefixes. Line segments guide the eye between completed prefix observations.
- `fixed-time.csv` evaluates a common budget for every method. Each fitted seed completes the floor of the budget divided by its measured median time, capped at the observed number of candidates. Each row gives all three counts. `fixed-time-paired.csv` subtracts the comparator recovery from exact-composition drift on each reference ID.
- `time-to-first-summary.csv` gives the cost to the first observed hit or the observed cap, averaged over all cases. It also reports median estimated time among recovered cases and the number censored in every fitted seed.
- Batch 64/256 tables divide the full occupied batch cost by its size, representing throughput across simultaneous requests. The serial figure uses batch one. Timings exclude retrospective matching and physical screening, so these quantities are estimated proposal costs.
- `independent-check.json` reconstructs every fixed-time estimate directly from the first-hit indices and original timings. The timing estimates and plots have separate provenance metadata.

## Crystal illustrations

```shell
python plot_crystals.py
python plot_crystals.py --unseen
```

The first four distinct formulas in the generated sequence fix the example IDs. Both states use the same physical scale, unit-cell chart, viewing angle and fractional origin convention. Circles have a common 0.51 Å display diameter. Cell-edge lines and nearest-neighbor connectors represent geometry. The panels have a 2 Å scale bar. The outcome figure includes all 32 originally selected candidates and the 23 with same-reduced-formula MP-20 references. The plots and a manifest with exact ordered example IDs and input hashes are under `../de-novo/figures/`.

The appendix montage uses the first four distinct reduced formulas absent from the restricted training set among the same 32. The selection reads the training-membership flag before accessing relaxation status or energies. Its metadata lists all four endpoint energy gaps and the available matched-reference gaps.

