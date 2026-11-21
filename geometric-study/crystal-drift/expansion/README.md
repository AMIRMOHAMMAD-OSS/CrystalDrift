# Reproducing the expanded study

Run commands from the repository directory containing `paper/main.tex`. The full research ZIP includes both the historical experiments and this expansion. The minimal arXiv ZIP contains only the TeX files and figures needed to compile the paper and supplement. See the [claim audit](CLAIM-AUDIT.md), [comparator protocols](BENCHMARKS.md) and [short change log](CHANGES.md).

## Hardware and dependencies

The compact networks were fitted on the local Intel Core i7-12700H using one PyTorch CPU thread per fit. Conditional and end-to-end de novo inference were timed on a GeForce RTX 3060 Laptop GPU under WSL, with one Torch host thread, FP32, and host output materialization. Each timed job held the GPU separately. Other CPU jobs shared the machine. The five-step external quality replay records its TF32 setting independently of the TF32-off common timing run. No configured Jean Zay allocation was available. The [scaled run guide](../scaled/RUN.md) documents the existing Slurm and Apple Silicon workflows. A Mac Studio can run the compact training and CPU analysis. It cannot reproduce the reported CUDA timing directly.

See the [training run guide](training/README.md), [de novo run guide](denovo/README.md) and [released CrystalFlow guide](external/README.md) for environment pins, weight identifiers and the exact commands. Most table reductions require only Python and NumPy/SciPy. The full matching stage used pymatgen 2024.4.13. The raw generation evaluator also requires SMACT and matminer, and the released MP-20 energy network requires the compatible PyTorch Geometric stack. Run physical screening in the CHGNet 0.3.8 environment with the 0.3.0 weights, ASE and SciPy. All pseudopotential files are saved with their hashes. Electronic-structure runs use the program version identified in each PWSCF output.

## Training, matching and timing

1. Use the compact-control guide for training, validation, frozen selection, generation, matching and GPU timing. It describes twelve training trajectories, each with three checkpoints. The 16-step and one-Euler rows use the same fitted flow weights. Individual validation scores, parameter and schedule hashes, ordered test geometries, timestamps and explicit failure masks are archived. The principal conditional selection uses only validation.
2. Use the external replay guide for the separate released conditional and unconditional CrystalFlow evaluations. Released one-step uFlowCSP uses the [pinned model and replay](../revision/full/external/README.md). Full MP-20 contains 9,046 targets. The CPU binary cohort, full-chemistry timing prefix and independent Alexandria cohort are distinct sets. The older three-seed WBM experiment and its training-only donors are in [the original protocol](../revision/full/README.md).
3. Use the de novo guide for generation, raw scoring, property scores, nearest fingerprints, uniqueness, novelty, the separate seen/unseen estimates and measured generation costs. The geometric-generator comparison uses identical chemistry. The joint-pipeline comparator uses all 10,000 ordered deposited CrystalFlow structures with the original failure position intact.

The new mathematical, hybrid and volume checks can also be run independently.

~~~sh
python expansion/math.py
python revision/ablation/checks.py
python expansion/hybrid/prepare.py
python expansion/hybrid/score.py --workers 4
python expansion/hybrid/paper.py
python expansion/volume/correction.py --workers 4
python expansion/volume/paper.py
~~~

In the hybrid experiment `prepare.py` requires the [source Alexandria preparation](../scaled/ALEX.md). To repeat matching without downloading the entire input corpus, use the archived 512-target cohort and ordered proposal banks with `score.py`. It also checks the original exploratory full-split crosswalk. The correction uses only official MP-20 training volumes and reruns the matcher on the reconstructed native cells.

## Physical screen and DFT

The five-arm [physical design](physical/DESIGN.md) specifies candidate order, phase selection, force and contact criteria, energetic threshold, novelty, denominator and illustration rules. The phase inputs are shipped, so a replay of their relaxed energies only needs the CHGNet environment. The following commands regenerate inputs from the source pools, finish missing physical records and rebuild the tables and images.

~~~sh
python expansion/physical/run.py --prepare empirical-drift=expansion/denovo/pools/empirical-drift.npz learned-drift=expansion/denovo/pools/learned-drift.npz learned-flow16=expansion/denovo/pools/learned-flow16.npz learned-uflow=expansion/denovo/pools/learned-uflow.npz crystalflow-joint=expansion/external/crystalflow-joint.npz
python expansion/physical/run.py --workers 3
python expansion/physical/summarize.py
python expansion/physical/render.py
python expansion/physical/paper.py
python expansion/dft/inputs.py
~~~

The [DFT selection](dft/README.md) fixes the first eligible shared formula and four coordinates. With a working Quantum ESPRESSO executable, the portable command below needs only the executable and the supplied pseudopotentials; run from Linux or WSL. The launcher writes a single-point output for every generator, geometry and k mesh. It resumes complete converged calculations. Energy and force comparisons use identical settings within a mesh.

~~~sh
QE_PW="$(command -v pw.x)" bash expansion/dft/run.sh
python expansion/dft/summary.py
~~~

The independent joint CrystalFlow replay uses a fresh 10,000-draw bank and a separately registered first sixteen. The [local physical-screen design and commands](physical-joint-local/README.md) describe candidate order, phase reuse and verification. To reproduce its DiffCSP metrics, run `python expansion/denovo/secondary.py --wait` in the evaluation environment as the joint sampler finishes. This secondary cohort has its own generation clock, chemical sample and relaxed candidates.

## Evidence and submission package

The old audit independently reconstructs the binary, WBM, novelty and ablation claims. The new audit checks the full conditional timing, hybrid cohorts, volume and de novo reductions, physical counts and electronic-structure outputs. Run the specialist verifiers described in the linked component guides first.

~~~sh
python revision/verify_claims.py
python expansion/training/verify.py
python expansion/external/seal.py expansion/external/csp-s5
python expansion/verify.py
python expansion/method_figure.py
python revision/package.py source --stage ../crytaldrift-clean-build
(cd ../crytaldrift-clean-build && pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex && pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex)
python revision/submission_check.py ../crytaldrift-clean-build
pdftocairo -svg paper/exp-structures.pdf revision/figures/exp-structures.svg
pdftocairo -svg paper/crystal-examples.pdf revision/figures/crystal-examples.svg
pdftocairo -svg paper/unseen-examples.pdf revision/figures/unseen-examples.svg
python revision/package.py figures
python revision/package.py crystals
python revision/package.py tables
python revision/package.py research
python expansion/seal_release.py
~~~

The build directory must be new. `revision/submission_check.py` compares every extracted member to the manuscript source, checks the compiler log, validates bibliography/figure counts, internal labels and linked bibliography URLs, verifies embedded fonts, and copies the resulting PDF into `paper/`. The final release provides SHA-256 digests for the PDF and the archives.
