# Formula-only training and inference

Run commands from the release directory that contains `CrytalDrift_overnight_goal.md`. The generator spelling in the paper is CrytalDrift. Reuse the release's training, CUDA and pinned matcher environments. The Windows paths and Linux interpreter are recorded in `../../v2/csp/local-runtime.json` relative to this directory. Use `MATCHER_PYTHON` to override the local matcher path. The APIs import with `PYTHONPATH` set to the release root.

## Seed law and eligible positives

`Sampler` intersects the registered protected training IDs with verified Hall round trips whose native cells are primitive. This leaves 26,419 rows and 24,310 distinct exact unit-cell formulas. Training excludes every row of the anchor's reduced chemical formula from both the symbolic shortlist and the shape/residual donor draw. All 105,676 primary stochastic training seeds passed an independent donor-ID recheck; the first traversal used a P1 fallback 544 times. The 26,419-row training bank supplies the chemical volume regression, pattern inventory, fallback shapes, reference geometries and the feature calibration. Calibration uses a fixed random subset of 256 such rows and independent allowed symbolic candidates. The same Laplacian or Gaussian bandwidth pair, 0.741924 and 1.978464, is applied to every arm. Each is a fixed multiple (0.3 or 0.8) of the training median standardized angular-feature distance from a seed to its observed structure. A block ablation never rescales a surviving block.

Within a stoichiometric count signature, order element roles by count, Mendeleev number and atomic number. Rank training analogues by mean Mendeleev distance plus a two-unit element-mismatch cost. Transfer unique sorted element-labelled Wyckoff occupations in rank order. The seed samples a compatible training shape preferentially at the same letter multiset, then the same group, then across eligible training. The cell is group-averaged in the conventional basis and scaled to the composition regression plus the allowed donor's residual. Sample independent uniform chart parameters, expand and primitive-reduce by conventional centering. Native sites retain chart Jacobians, labels and exact unit-cell stoichiometry. This implementation samples the continuous distribution once and retains any specialization or short contact for evaluation. The unfitted zero-call prior and every symbolic refinement share precisely the same candidate arrays. At inference the outer loop uses distinct shortlist entries in order and cycles if fewer entries than requested exist. The train draws choose independently from the first five eligible entries with replacement, so their marginal seed law covers the primary five-draw endpoint but the fixed order and dependence across a target's proposal prefix differ.

## One evaluation

The input is the sampled lattice, all sites, atomic numbers and orbit membership/Jacobians from the inferred Hall template. Enumerate all directed images inside 4 angstroms, excluding the central self-image. Each center pools element embeddings and smooth radial messages. Average these descriptors over equivalent sites. Each occupied-orbit token then receives a multiplicity-weighted global pooled state in each of three (small) or four (medium) residual MLP blocks. Time inputs are fixed at zero and one for drift sampling. A bounded scalar edge weight multiplies each physical edge vector. Convert the resulting site velocity field to each orbit's independent chart update by Cartesian least squares on all its images. Analytically expand this one chart update at all sites of the orbit. The cell head uses a physical sum of edge dyads normalized by site count. Its symmetric matrix update is positive definite since the spectral norm of the bounded dyad sum is at most one. Normalize its determinant, then multiply by the separately learned positive volume ratio to the one-third power. Metric and origin-transport statements and numerical property checks are given in the geometry supplement. The deployed one-call map is equivariant for an appropriately transformed chart/seed tuple. The shortlist uses one conventional Hall setting.

The small and medium networks instantiate 218,497 and 2,098,297 registered parameters. All learned modules are part of the deployed forward evaluation. With drift, the first time input is constantly zero, so its encoder column has no gradient; the unit-valued second time column can be absorbed into a bias. The equivalent MeanFlow network varies both times during training. An element embedding row that never occurs on training also receives no data gradient.

## Field, training exposure and MeanFlow

Every optimizer update uses 16 scheduled anchors (except the last partial minibatch and size-profile regimes). One eligible anchor and one independently resampled structure of its exact formula give two positive references; two negative seeds and two live queries are generated independently conditional on that formula. Positives and generated structures use native-cell geometry with their own carried orbit memberships. The four queries/negative states come from the same frozen seed bank for all primary arms. Updates use the stopped finite-bank objective `v3/geometry/descriptors.py`. Laplacian and Gaussian affinities both apply row/column normalization. Feature differences weighted by the opposite bank mass define the displacement; the two normalized temperature contributions are averaged. A scheduled epoch traverses each eligible ID once. The first positive's position in this without-replacement schedule differs from i.i.d. empirical-bank sampling. Zero expected drift therefore only follows under the separate joint bank-swap hypothesis in the supplement. Singletons occur for 22,916 scheduled anchors. The independent query/negative generator seeds, frozen for matched comparisons, do not create more independent positive data at those formulas.

All one-traversal arms use the same 16-anchor minibatch, AdamW, a 16-update warmup to 0.0003, gradient clipping at norm 2, and float32 throughout. Checkpoints after 103/207 shards contain 13,184/26,419 distinct positive anchors. Both are scored at five ordered draws over 96 development targets. Equal match counts choose the earlier checkpoint. A separate 96-target confirmation panel receives only the frozen selection. Additional training exposures and fresh banks are registered in `EXTENDED.md`. Each fit saves atomic snapshots, the optimizer/RNG for restart, SHA-256 digests and an updating training trajectory. Timings in fit receipts include milestone validation prediction, checkpoint writes and initialization after the training loop timer starts. They are observed accelerator occupation times under stated concurrent host load.

For the same-backbone MeanFlow control, all weights, symbolic inputs, two active seeds, target structures and data pass are matched. For each real/seed pair, assign equal-species sites using the Hungarian solution for wrapped differences at the chosen reference origin. Project the assigned chord to the seed orbit tangent. Project the reference metric into the seed group by rotation averaging, renormalize its determinant to reference volume, and transport the result by the positive square root of its relative metric. Interpolate linearly from the resulting compatible target (time zero) to the seed (time one). The predictor returns a lattice increment and an unwrapped fractional increment pointing toward data. Its training target is minus the path velocity minus the interval times the total derivative of this signed predictor. The derivative is a stopped, exact reverse-over-reverse autograd JVP along the path. Uniform independent event times are sorted; 30% have both times set equal. Mean squared cell and physical site errors are normalized by 4 angstroms, and averaged. The objective is a same-backbone control of the MeanFlow identity in a template-constrained chart. Its canonical matching/projection training coupling depends on the chosen representative. The external uFlowCSP comparison uses its separate published input representation, conditioning and native prior. [Geng et al.](https://arxiv.org/abs/2505.13447) [uFlowCSP](https://arxiv.org/abs/2609.09799)

## Reproduction

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m lattice_symmetry.v3.models.prepare
# On the local WSL installation, append the compatible Numba dependency site before importing model code.
export CRYTALDRIFT_NUMBA_SITE=/home/amir/miniconda3/envs/scrna/lib/python3.11/site-packages
/home/amir/miniconda3/envs/mp_lstm/bin/python -u lattice_symmetry/v3/models/gpu.py --tier small --arm angular
/home/amir/miniconda3/envs/mp_lstm/bin/python -u lattice_symmetry/v3/models/gpu.py --tier small --arm contacts
/home/amir/miniconda3/envs/mp_lstm/bin/python -u lattice_symmetry/v3/models/gpu.py --tier small --arm reciprocal
/home/amir/miniconda3/envs/mp_lstm/bin/python -u lattice_symmetry/v3/models/gpu.py --tier medium --arm meanflow
python -m lattice_symmetry.v3.models.select_weights --confirm
python -m lattice_symmetry.v3.models.audit_training
python -m lattice_symmetry.v3.models.meanflow_test
python -m lattice_symmetry.v3.models.diagnose
```

For independent re-training runs use a fresh evidence extraction without the corresponding `fits` directory. For resumption keep its latest `resume.pt`. `gpu_window.sh` records the original launch; `window.py` schedules the complete run matrix. The single-process supervisor takes `--coordinated` only for the shared research session to release the accelerator for a comparator and separate clock; omit this flag for unattended reproduction in a dedicated machine. For the registered longer exposure run the same entry point with `--passes 8` after its base one-pass checkpoint. The deterministic extra banks can be rebuilt in RAM shard-by-shard; `extra_banks.py --workers 1 --part 0` optionally caches them on disk. `long_selection.py` applies the locked four-milestone selection and performs the frozen confirmation. On authorized Slurm hardware replace the Python path with the activated training environment and run these exact modules inside the allocated GPU job. The mandatory model/geometry code needs no network or cluster login.

The rotated local CUDA runs use the following additional settings. The later bank archive combines shards already present in the workspace with WSL scratch shards. Their hashes are saved by the CPU producer, and seeds and row order in `prepare.py` suffice to reconstruct each shard. For a different installation set the two scratch paths to existing writable space on the training host. Set `MATCHER_PYTHON` to its pinned evaluator interpreter when reproducing outside the documented Windows/WSL installation.

```sh
export CRYTALDRIFT_BANK_SCRATCH=/tmp/crytaldrift-v3-20261121/banks
export CRYTALDRIFT_SCRATCH=/tmp/crytaldrift-v3-20261121/checkpoints
python -m lattice_symmetry.v3.models.extra_banks --workers 1 --part 0 --scratch-dir "$CRYTALDRIFT_BANK_SCRATCH"
python -m lattice_symmetry.v3.models.radial_resolution
python -m lattice_symmetry.v3.models.window
python -m lattice_symmetry.v3.models.report_validation
python -m lattice_symmetry.v3.models.learning_curves
python -m lattice_symmetry.v3.models.fixed_cost
```

When `CRYTALDRIFT_SCRATCH` is set, later optimizer states and the new medium/large deployable tensors live in that directory. Each milestone's `.pt.pointer.json` identifies the actual file, byte count, SHA-256, Linux path and WSL UNC path. `Sampler.load` follows the receipt automatically on the original machine. To move the study, copy the identified file to the ordinary `.pt` location beside its receipt, and verify the SHA-256. The small weights and the medium models trained before the scratch rotation are ordinary `.pt` files in their own directories. The comparator-scale profile writes training time, peak CUDA allocation and an estimate for the 13,210-update full traversal at microbatch two. A separately completed full training job writes deployable weights. The durable per-run receipts identify the storage choice for each fit.

All four eight-traversal seed-11 comparisons revisited each of the 26,419 eligible anchors eight times. The best development checkpoint was the half-pass checkpoint in each row. Both panels contain 96 targets and use five ordered draws.

| Model | Half pass | One pass | Four passes | Eight passes | Frozen confirmation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small angular | 63 | 63 | 62 | 62 | 50 |
| Medium angular | 62 | 62 | 62 | 62 | 51 |
| Medium contact | 63 | 63 | 62 | 62 | 50 |
| Medium MeanFlow with matched order | 63 | 63 | 61 | 63 | 51 |

Reproduce the three-seed validation contrasts with `python -m lattice_symmetry.v3.models.replications` and the hardware/exposure table with `python -m lattice_symmetry.v3.models.learning_curves`.

