# Released uFlowCSP replay on the full MP-20 test split

The evaluation conditions on the species and integer site counts of each of the 9,046 official MP-20 test records. The published Cartesian uFlowCSP checkpoint generates up to 20 unrelaxed structures per record in one network evaluation per structure. The records match the authors' complete test ID list and are disjoint from their training IDs. Each target receives the null space-group input. The network, chemistry table and preprocessing follow the earlier 128-target replay. [Pinned Cartesian implementation](https://github.com/usccolumbia/uFlowCSP/tree/e87af8483e6d45b0217b4ede430862e992d2b1d0) · [Released checkpoint](https://doi.org/10.6084/m9.figshare.33974422.v1) · [Earlier replay and provenance](../../../uflowcsp-results/README.md)

## Generation

Targets are sorted by the SHA256 digest of the material ID. The first five predictions for the first 128 IDs are copied from the previously scored CPU replay. The remaining predictions use the same Cartesian prior, EMA weights and decoding with PyTorch 2.6.0+cu124, timm 1.0.21 and FP32 arithmetic on an RTX 3060 Laptop GPU. A minibatch starts at global index `i` with zero-based sample index `d` and seed `2026000000 + 1000000*d + i`. Each resumable 256-target tile uses consecutive minibatches of 24 and a shorter last minibatch. The 128-target tile forms its own group. The prior runs on CPU before moving to the GPU. All 100 padded tokens enter the transformer. Truncating the same prior to 20 tokens on the first eight targets changed live output components by a mean absolute 0.8903 and maximum 14.0904. The generated pools use all 100 tokens. Output decoding averages the live lattice predictions, saves fractional coordinates in ascending atomic-number order, and writes the six coordinates of the symmetric log stretch.

`generate.py` verifies the Figshare-derived EMA safetensors SHA256, all four source-file Git blob hashes and a strict load of all 141 state tensors. `prepare.py` writes the explicit non-persistent chemistry features and ordering from the pinned resources. The checkpoint has 131,215,116 trainable weights. The manifest records exact versions, source hashes, GPU and the token check. Each output tile records its draw and target index, IDs, composition, geometry and wall time including generation and encoding through the start of serialization. The archived CPU prefix has copy times in these fields. They cannot support a common-hardware speed comparison.

## Recovery

Raw coordinates go directly to the same pymatgen 2024.4.13 matcher as the compact arms. It uses primitive reduction and equal-volume rescaling, a length tolerance of 0.3, a site tolerance of 0.5, an angle tolerance of 10 degrees, equal species, and disables supercell search and subset matching. We score finite normalized RMS from `get_rms_dist`, maximum-site success from `fit`, and same-candidate RMS success with native volume within a factor of 1.10 of the target. Native volume uses the absolute log volume ratio prior to the matcher's rescaling. The conservative candidate screen and accelerated matcher receive additional sampled cross-checks against both public methods. A raised disagreement terminates the run. The score tiles retain the raw RMS, strict success and native log ratio for every candidate.

`report.py` reconstructs each target's first 1, 5 and 20 candidates, tests that the 128-by-five input geometry is exactly preserved and compares its raw RMS outcomes with the earlier independent match calls. Confidence intervals resample material IDs with replacement 10,000 times, preserving method pairing. Compact-model results average three fitted seeds. The released model contributes one checkpoint and inference seed. Normalized best RMS is averaged over recovered targets within each fitted seed, followed by averaging across those seeds.

The files `mp20/full-k{1,5,20}.json` and their associated case CSVs contain completed full-cohort results. The corresponding `prefix` files concern 128 targets. The JSON includes case counts, intervals, candidate counts, parity checks, exceptions and paired compact-minus-released contrasts once the compact scores cover the same budget. `mp20/progress.json` can contain partial denominators while a run is active.

From the crystal-drift directory run the following, using the environment carrying the pinned dependencies for each stage.

```sh
python uflowcsp_weights.py
python revision/full/external/prepare.py
python revision/full/external/generate.py --weights /path/to/ema.safetensors --batch 24
python revision/full/external/match.py --workers 2
python revision/full/external/report.py
python revision/full/external/seal.py
python revision/full/external/paper.py
```

The shared full-split compact evaluator creates `revision/full/mp20/cohort.npz` and the compact score tiles used for pairwise contrasts. The released model consumes the original 20-slot test arrays and the earlier 128-target saved prediction pool. The model source files, author train/test IDs and checkpoint extraction instructions are in the earlier replay directory. The matcher can score late proposals in blocks of five as soon as they are generated and joins those results by target ID. The sealed `mp20/complete-crosswalk-k20.npz` gives every candidate's normalized RMS, maximum-site decision and native log volume ratio in target-by-draw order. The accompanying `complete-manifest.json` verifies all 720 geometry tiles and 108 final score tiles by SHA256. `paper.py` writes the manuscript table directly from complete full-split summaries.
