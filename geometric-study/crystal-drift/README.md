# CrytalDrift

CrytalDrift generates a cell metric and all fractional positions in one neural evaluation. The study benchmarks prediction from exact cell composition, then couples the geometry generator to fitted composition priors for full multielement MP-20 generation.

[Manuscript and mathematical supplement](paper/main.pdf) · [arXiv source ZIP](paper/crystal-drift-arxiv.zip) · [Reproducing the expanded study](expansion/README.md) · [Claim audit](expansion/CLAIM-AUDIT.md) · [Change log](expansion/CHANGES.md) · [Generated and relaxed multielement cells](expansion/physical/structures.png)

The main comparisons use 9,046 official MP-20 test structures, same-hardware serial GPU deadlines and five unfiltered 10,000-proposal de novo pools. An independent sixth pool replays the released joint CrystalFlow generator and pairs its measured cost with 10,000 locally scored draws and sixteen preselected relaxation cases. The binary CPU study, three-seed WBM models, independent Alexandria hybrid test, descriptor collision and matched feature ablation retain their original cohorts. [Expanded evidence index](expansion/CLAIM-AUDIT.md) · [Original evidence index](revision/EVIDENCE.md)

## Run the model



These commands reproduce the original binary model. Run them from this directory. Core generation used Python 3.10.8, NumPy 1.26.4, SciPy 1.10.1 and PyTorch 2.7.1 on CPU. Matplotlib 3.7.1 renders the figures. The saved CPU checkpoints can be sampled without retraining.



~~~sh

python sample.py Mg2Zn4 --count 5 --seed 714 --out samples-Mg2Zn4

python de_novo.py --out new-de-novo

~~~



The first command fixes the actual atom counts in a unit cell. The second draws 1,024 compositions with both elements and multiplicities determined by the training distribution, generates all their coordinates, and exports the first 32 as P1 CIFs. The default neural checkpoint for both commands is `drift-results/multi-exact/seed11.pt`. The composition prior and checkpoint are distinct parts of the de novo model.



## Reproduce the comparisons



The six processed arrays in `data/` suffice for fitting and geometric evaluation. For extraction from raw CIFs or replay of reference selection, place the three original compressed split files in `data/` and run `python data.py`. The complete research archive supplies these source files. The manifest records the authors' download URLs and file hashes. [Split manifest](data/manifest.json)



~~~sh

python verify.py

python train.py

python train.py --validation-only --methods ab ab-exact multi multi-exact

python baseline.py

python baseline_multi.py

python compare.py

python verify_comparison.py

python standard_match.py --external

python external_comparison.py --require-match

python speed.py --export --device cpu

python frontier.py

python time_frontier.py

python verify_frontier.py

~~~



Run `python speed.py --device cuda` after CPU export to benchmark a CUDA-enabled installation. The supplied CPU/GPU observations use different operating systems and PyTorch versions. Rebuilding the prefix plots reads the two timing files. The matching environment used pymatgen 2024.4.13 and spglib 2.5.0. The external uFlowCSP predictions are bundled. Their 131-million-parameter checkpoint can be obtained separately by following the replay guide. [uFlowCSP replay](uflowcsp-results/README.md) · [StructureMatcher outputs](structure-matching/by-method.csv) · [Timing observations](speed-results/)



## Examine the physical screen



The de novo stream is frozen in `de-novo/all-draws.npz`. Every original draw is listed in `de-novo/manifest.csv`. The first 32 received the independent physical screen. Candidate IDs link the original and relaxed CIFs, forces, contact distances, energies and illustrated examples. The reference selection and stopping rules are documented separately. [Screening report and environment](de-novo/relaxation/README.md) · [Per-structure results](de-novo/relaxation/results.csv) · [Elemental comparisons](de-novo/elemental-screen/candidate-gaps.csv)



~~~sh

python relax_candidates.py

python elemental_screen.py

python plot_crystals.py

python plot_crystals.py --unseen

python time_de_novo.py

~~~



The following figures apply to the binary experiment. At a serial 5 ms CPU proposal budget the exact-composition model recovered 43.75% of the 128 held-out targets. At five proposals each, the released one-step uFlowCSP model recovered 80.47%. CHGNet relaxation of a generated cell took a median 3.12 seconds and left seven of the 32 with a contact below 60% of summed covalent radii. Physical viability requires evidence beyond the potential-based screen.



## Repeat timing on Jean Zay



The [Slurm job](jean_zay_crystal.slurm) uses the bundled checkpoints by default. From this directory on Jean Zay, substitute the H100 allocation name for PROJECT in `sbatch --account=PROJECT@h100 jean_zay_crystal.slurm`. Appending `retrain` refits the compact drift and flow models for both cohorts and all three seeds. Each run writes its hardware metadata and raw CPU/GPU timings in a separate job-results directory keyed by the Slurm job ID. The script passed local Bash syntax checking. There was no configured Jean Zay connection or allocation in this session. All reported hardware results were measured locally.



## Build the manuscript



The arXiv source archive contains the TeX and vector figures with no external file dependencies. From its root run `pdflatex main.tex` twice. The separate research archive contains the model, checkpoints, source-data attribution, held-out predictions, per-sample evidence and plotting code.

