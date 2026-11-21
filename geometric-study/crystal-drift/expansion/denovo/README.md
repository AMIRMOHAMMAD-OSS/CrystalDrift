# Multielement MP-20 de novo benchmark

The evaluation bank contains 10,000 unfiltered draws per arm. The train-only empirical prior samples one of 27,136 training compositions. The autoregressive prior draws an atomic-number-ordered multiplicity pattern and then distinct species using smoothed training transition counts. Validation selected shrinkage 4. Sampling uses neither a test composition nor a reference structure. The learned-prior arms receive byte-identical compositions, and both compact models share the same input noise and one pass of training. The distinct validation-selected flow in the composition-conditioned study used ten passes.

The compact generator weights are `scaled/runs/mp20/{drift,flow}-seed11.pt`. Exact checksums and seeds appear in `pools/*.json`, and `prior-fit.json` records the training archive hash, candidate hyperparameters and validation support. `pools/*.npz` contain ordered draw IDs, atomic numbers, six native lattice coordinates, the decoded lattice and fractional positions. Cartesian positions are the product of the latter two arrays. The native joint baseline is the output of `expansion/external/import_joint.py`, with its download and attribution in [the external benchmark](../external/README.md). Draw 4731 in zero-based order has an empty POSCAR and remains in the denominator.

## Reproduce from the repository root

The CPU scoring environment used Python 3.10, numpy 1.26.4, scipy 1.10.1, pymatgen 2024.4.13, spglib 2.5.0, matminer 0.9.2, SMACT 2.5.5 and numba 0.59.1. The compact generation and property runs used PyTorch 2.7.1 on CPU. The GPU timing environment used PyTorch 2.6.0+cu124 and timm 1.0.21 on an RTX 3060 Laptop GPU. The uFlowCSP weights and authors' source are prepared following [its replay instructions](../../revision/full/external/README.md).

~~~sh
python expansion/denovo/generate.py --device cpu --batch 128
python expansion/denovo/uflow.py --weights /path/to/ema.safetensors --batch 24
python expansion/denovo/gpu_timing.py
python expansion/denovo/gpu_timing.py --match-noise
python expansion/denovo/uflow.py --replay --weights /path/to/ema.safetensors --batch 24
~~~

The last two commands require CUDA. Run the following after importing the CrystalFlow release. To reproduce the original CPU evaluator literally, omit `--compiled`. The optional compiled kernels numerically implement the pinned pymatgen angular and Voronoi-face formulas. The parity script runs stock first and then checks all per-site feature components at 12 randomly chosen draw positions per arm, including held-out references. `fp-parity.json` gives the 72 positions and the maximum errors. The benchmark combines stock and parity-tested compiled tiles.

~~~sh
arms="empirical-drift learned-drift learned-flow16 learned-uflow crystalflow-joint"
python expansion/denovo/score.py --kind raw --chunk 100 --workers 4 --arms reference $arms
python expansion/denovo/fast_ops.py
python expansion/denovo/verify_fast.py --n 12
python expansion/denovo/verify_fast.py --canonical --n 4
python expansion/denovo/score.py --kind fp --compiled --chunk 20 --workers 6 --arms reference $arms
python expansion/denovo/novelty.py --workers 4 $arms
python expansion/denovo/report.py
~~~

The property evaluator uses the authors' released MP-20 DimeNet++ weights, lattice graph construction, periodic triplets and formation-energy scaler. The fetcher verifies every file against the authors' Git blob at commit `7121d159826efa2ba9500bf299250d96da37f146`, and the recorded SHA-256 hashes are checked before inference. Install torch-geometric 2.6.1 plus torch-scatter and torch-sparse wheels that match the installed torch version. The reference CPU run used torch-scatter 2.1.2+pt27cpu and torch-sparse 0.6.18+pt27cpu. The inference batch size is 32. Model outputs are saved in eV/atom after inverse scaling. [Released DiffCSP evaluation code](https://github.com/jiaor17/DiffCSP/tree/7121d159826efa2ba9500bf299250d96da37f146)

~~~sh
python expansion/denovo/fetch_predictor.py
python expansion/denovo/property_graphs.py --workers 4 reference $arms
python expansion/denovo/property_eval.py reference $arms
python expansion/denovo/report.py
python expansion/denovo/write_paper.py
python expansion/denovo/validate.py
~~~

## Records and denominators

- `scores/raw` contains each proposal's chemistry, 0.5 Å distinct-site and minimum-volume test, crystallographic density, element count and any failure. `scores/fp` contains site-averaged 61-component CrystalNN and 132-component Magpie vectors and featurization errors.
- `panel/*-cases.npz` retains all 10,000 raw IDs in draw order. `valid` includes successful fingerprint calculation when all fingerprint tiles are present. `w1_sample` lists exactly 1,000 indices sampled without replacement with NumPy RandomState seed 7121. Density, formation-energy and element-count distances all use these same rows against all 9,046 references. `energies` records each ID and predicted energy. The report checks those identities before introducing a property claim.
- `panel/*-coverage.npz` contains separate nearest-neighbour distances in standardized composition and raw structure space. Precision counts successful generated rows within both cutoffs over 10,000 proposals. Recall counts held-out reference rows within both cutoffs over 9,046 references. A proposal can pass coverage using different neighbours in the two spaces.
- `novelty/*.csv` records the earliest matching representative and the first matching training ID in the same reduced-formula group. The matcher uses primitive cells, volume scaling and element identity with lattice/site/angular tolerances 0.3/0.5/10 degrees. Uniqueness is an ordered greedy count. The training archive has 27,136 structures. The empty CrystalFlow record counts toward the denominator but contributes to no success count.
- `panel/*.json` records all numerator counts, denominators, fingerprint completion flags, raw density and geometry broken out by seen/unseen reduced formula. `panel/paired-validity.json` compares the shared-composition geometry arms with a joint draw-level bootstrap, conditional on the trained models. Training-seed uncertainty for the conditional experiments is separate.
- GPU generation times include sampling compositions, sampling geometry noise, network execution, transfers and decoding native lattices and sites, with disk writes outside the timer. Local timing used batch 24 and 128 warm batch-one draws. Valid unique cells per minute multiply the evaluation-bank fraction by the measured generation rate. The primary replay recreates the exact random variates of the quality bank by emitting lattice and site noise in the original order before batching. Per-row geometric deviations are in `pools/*-gpu-parity.npz`. The initial compact measurements used an independent draw of the same geometry noise law because changing the inference batch size changes how NumPy interleaves lattice and site variates. Both measurements and their runtime differences are retained. The scored uFlowCSP bank is the output of its initial timed GPU block.
- `panel/audit.json` independently reconciles ordered examples, raw validity and novelty masks, W1 property IDs, held-out energy sanity and coverage reductions for all arms. `manifest.json` identifies scripts, raw outputs, predictor, split archives and checkpoints by SHA-256.
- No stability criterion enters these raw metrics. Formation-energy distribution distance uses a learned predictor. The separately timed physical relaxation records are in the physical-screening portion of the study.

## Independent joint-sampler replay

The deposited CrystalFlow sequence is the fifth primary arm. A fresh 100-step joint CrystalFlow run, when available, is a distinct secondary experiment. The primary comparators and the five-arm relaxation screen retain their original draws. After producing `pools/crystalflow-joint-local.npz` with `expansion/external/joint_timing.py`, run `python expansion/denovo/secondary.py --fingerprint-workers 5` in an environment containing the scoring and property dependencies. The program preserves the fresh bank's original order, recomputes chemistry, geometry, novelty, coverage and property metrics, and validates the individual records. The GPU and licensed checkpoint acquisition are described in [the external benchmark](../external/README.md).

Draw 5189 (zero-based) in the independent bank contains americium. Its atomic number 95 exceeds the maximum index 94 of the released formation-energy encoder's 95-row embedding. It remains in the raw and structural benchmark. The fixed 1,000-valid property sample contains no such draw and required no substitution.
