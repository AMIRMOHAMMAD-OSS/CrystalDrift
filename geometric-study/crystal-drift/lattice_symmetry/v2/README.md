# Reproducing one-step generation in periodic crystal geometry

The author uses **CrytalDrift**. This directory contains the periodic model and the supplied-template extension. The 38-page paper at `paper/main.pdf` in the repository root is the earlier fixed-basis study. Its full-MP-20, WBM, Alexandria, empirical-prior, multielement, CrystalFlow and electronic-structure experiments retain their original checkpoint identities. See its [study and reproduction guide](../../expansion/README.md).

## Files for review

The `release` directory contains the manuscript with its supplement, a self-contained source ZIP, editable SVGs, CIFs, machine-readable tables, and the research evidence ZIP. The source ZIP compiles independently. For a full numerical replay, extract `crystal-drift-research.zip` and then `CrytalDrift-periodic-evidence.zip` into the same parent directory. Both carry a top-level `crystal-drift/` directory. The former supplies the official MP-20 archives, earlier checkpoints, external-model evaluations, pinned matcher, property evaluator and competing phases. Its SHA-256 is `644c0456a4f015edb1df00ab766246a1c4922f692bb9329368cd7451c5ec84ec`. Keep the downloaded fixed-basis research ZIP next to the extracted crystal-drift directory so the baseline audit can hash the original archive. The overlay includes its companion archive digests and the three smaller fixed-release bundles required by the audit, as well as the periodic study. Run all commands below from the common `crystal-drift` directory.

- [Claim audit](CLAIM-AUDIT.md) and [source guide](sources/README.md). These map the manuscript to proofs, code, per-case records and original publications.
- [Formula-conditioned fit and scoring](csp/README.md) and [prespecified design](csp/DESIGN.md). These include the sources and commands for training, scoring, timing and replay. `csp/selection.json` names every frozen training checkpoint by seed and hash. `csp/test/*/crosswalk.npz`, `k.npy` and `f.npy` retain each target's ordered coordinates. `outcomes.npz` gives per-draw outcomes and exception masks.
- [Sampling and screening without test compositions](denovo/README.md) and [prospective rules](denovo/DESIGN.md). `denovo/pools` holds the actual 10,000-draw sequences and checkpoint receipts. `panel`, `energies`, `novelty` and `screen` retain the raw fingerprints, metric inputs, competing phases, checks, and generated and minimized CIFs.
- [Occupied-template fits and external-weight replay](wyckoff/README.md). `wyckoff/full` covers all official test IDs at one draw. `pools` and `scores` give each of the six methods' separate draw orders on the predeclared 128-template comparison. `operation-order-audit.json` reconciles operation lists, group types, and relaxed geometries.
- [Bounded-strain validation sensitivity](wide/CONFIRM.md). The extra shape update was evaluated on validation only. All main test results use the primary head.

## Hardware

Compact model fitting used the local Intel Core i7-12700H CPU with one PyTorch worker thread per process. Matching can use multiple independent one-thread processes. The common GPU clocks ran under WSL2 with PyTorch 2.6.0+cu124 on one GeForce RTX 3060 Laptop GPU. The timing receipts include dtype, batch size, host materialization, warm-ups, call counts, timestamps, versions and checkpoint identities. The native DiffCSP++ replay records its own TF32 and sampler settings. Concurrent CPU workload varied. The public prefix and balanced-order timing experiments retain separate results.

No configured Jean Zay login/allocation was available. The compact training, raw-data reductions and CPU timing can run on a Mac Studio with a compatible Python environment. CUDA measurements require an NVIDIA device. An Apple timing run must carry its own hardware label. The [larger-dataset guide](../../scaled/RUN.md) gives CPU and Slurm setup for the earlier models. For these periodic fits the monitored CPU jobs were sufficient. Using a clean copy with the data, the following runs the matched fit and scoring with installed dependencies.

~~~sh
export MATCHER_PYTHON=/path/to/pinned-matcher/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python lattice_symmetry/v2/csp/runner.py --matcher-python "$MATCHER_PYTHON" --workers 2
~~~

One possible Slurm invocation from that same directory on an authorized cluster is below. Set `JZ_ACCOUNT` to the allocation issued to the submitting user and activate the installations documented in the three guides. The first line requests CPU work only. The second requests one GPU for the three end-to-end de novo clocks. Both scripts have also been executed locally through their normal entry points. Cluster time depends on its hardware and filesystem.

~~~sh
sbatch --account="$JZ_ACCOUNT" --cpus-per-task=4 --mem=16G --time=12:00:00 --wrap='python lattice_symmetry/v2/csp/runner.py --matcher-python "$MATCHER_PYTHON" --workers 3'
sbatch --account="$JZ_ACCOUNT" --gres=gpu:1 --cpus-per-task=4 --mem=12G --time=01:00:00 --wrap='python lattice_symmetry/v2/denovo/gpu_clock.py'
~~~

The CPU fitting environment used NumPy 1.26.4, PyTorch 2.7.1 and Numba 0.59.1. Pin pymatgen 2024.4.13 and spglib 2.5.0 for matching and template analysis. See the component guides for CrystalNN, SMACT, the released DimeNet++ predictor and CHGNet 0.3.8. The native metric accelerator is compared against the stock public matcher. Computing scores from the stored arrays avoids a new training run. Training from scratch belongs in a fresh working copy because the drivers treat existing sealed milestones as resumable state.

## Additional clocks and the geometry sensitivity

Use the CPU Torch environment for the first command, and a CUDA-enabled environment with working Numba and SciPy for the next two. The CPU periods in the hardware files use one host thread. Run each timing job separately on the named device.

~~~sh
python lattice_symmetry/v2/csp/cpu_clock.py
python lattice_symmetry/v2/csp/interleave.py
python lattice_symmetry/v2/denovo/gpu_clock.py
"$MATCHER_PYTHON" lattice_symmetry/v2/csp/interleave_score.py
"$MATCHER_PYTHON" lattice_symmetry/v2/csp/sensitivities.py
~~~

## Checks and clean build

~~~sh
python lattice_symmetry/v2/protect_baseline.py
python lattice_symmetry/v2/acceptance.py
python lattice_symmetry/v2/checkpoint_acceptance.py --selected
"$MATCHER_PYTHON" lattice_symmetry/v2/csp/verify.py
python lattice_symmetry/v2/csp/report.py
python lattice_symmetry/v2/csp/training_figure.py
python lattice_symmetry/v2/csp/paper.py
"$MATCHER_PYTHON" lattice_symmetry/v2/wyckoff/verify.py
"$MATCHER_PYTHON" lattice_symmetry/v2/denovo/verify.py
"$MATCHER_PYTHON" lattice_symmetry/v2/denovo/analyze.py
python lattice_symmetry/v2/denovo/context.py
python lattice_symmetry/v2/denovo/clock_paper.py
"$MATCHER_PYTHON" lattice_symmetry/v2/denovo/render.py
python lattice_symmetry/v2/verify_study.py
python lattice_symmetry/v2/sources/bibliography.py
python lattice_symmetry/v2/release.py source --stage ../periodic-clean
(cd ../periodic-clean && pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex && pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex)
python lattice_symmetry/v2/release.py verify --stage ../periodic-clean
python lattice_symmetry/v2/release.py extras
python lattice_symmetry/v2/release.py evidence
python lattice_symmetry/v2/release.py seal
~~~

The build directory must be fresh. Verify the released archive digests against `release/SHA256SUMS.txt`. The manuscript verifier checks membership against the extracted source, external reference targets, all citation and label keys, embedded fonts, page text and the order of the references and supplement. `release/evidence-inventory.json` identifies each overlay member. The release stops at an author review package.

For an independent replay into a new directory, run the following from the repository root. It checksums every overlay member after extraction, runs the target-level audit there, and verifies that the regenerated numerical report has exactly its released hash.

~~~sh
python lattice_symmetry/v2/replay_archive.py crystal-drift-research.zip lattice_symmetry/v2/release/CrytalDrift-periodic-evidence.zip ../periodic-evidence-replay
~~~
