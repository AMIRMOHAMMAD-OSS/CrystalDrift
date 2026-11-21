# Multielement crystal generation on MP-20 and WBM

The 20 November revision adds the complete 9,046-ID evaluation and three native WBM fits. Its commands, paired results and software versions are in the [revision guide](../revision/full/README.md). The 128-case tables below document the original runs.


Both fitted models predict a cell and every active fractional position from an exact list of elements and atom counts. Their 24,377-parameter permutation-equivariant networks share the same width, masked pooling and learned element embeddings. Each site sees the periodic phase of its noise relative to both its own species and the whole cell. Drift makes one network call. The comparator learns a velocity for a componentwise wrapped and then mean-free interpolation on fractional sites and uses one or sixteen Euler calls. The one-call model trains with species-resolved Fourier power and cross-correlations between each species and two reference densities, the whole cell and its first species in ascending atomic-number order. This finite feature set can leave distinct structures indistinguishable. These files support elemental through seven-species MP-20 structures and arbitrary element multiplicities up to twenty occupied sites. No target coordinates, symmetry or material identifiers enter the generator.

## Data and splitting

| Dataset | Training | Validation | Test | Selection |
|---|---:|---:|---:|---|
| MP-20 | 27,136 | 9,047 | 9,046 | All released records. Published splits and native P1 cell bases. |
| WBM | 205,091 | 25,117 | 26,051 | DFT-relaxed native cells with at most 20 sites. Of 256,963 records, 704 exceed the size limit. Reduced formulas are disjoint across splits. |

MP-20 training alone includes 21,386 ternary or higher-species structures. Its authors' official split permits shared formulas. The WBM grouping is a new generative-model partition of the released relaxed structures. Matbench Discovery uses WBM for a different stability-prediction benchmark. Consequently, a generator trained here on WBM is unsuitable as an independent WBM stability benchmark entry. Repeated structures or prototypes across different reduced formulas remain possible. `audit.py` records the overlap for full partitions and fixed prediction cohorts. [MP-20 source and method](https://github.com/txie-93/cdvae/tree/main/data/mp_20) · [WyckoffDiff release](https://github.com/httk/wyckoffdiff) · [WBM benchmark and data](https://doi.org/10.1038/s42256-025-01055-1) · [DFT-relaxed WBM records](https://doi.org/10.6084/m9.figshare.22715158)

## Local reproduction, including macOS

Run these commands in `crystal-drift/scaled`. On an Apple Silicon Mac with Python 3.11 installed, the following creates a native CPU environment. On an Intel Mac with Python 3.11, replace the PyTorch version with `torch==2.2.2`, which has an x86-64 wheel. The scored local runs used Windows Python 3.10.8, NumPy 1.26.4 and PyTorch 2.7.1 CPU. A macOS execution has not been measured. [Apple Silicon package](https://pypi.org/project/torch/2.7.1/#files) · [Intel Mac package](https://pypi.org/project/torch/2.2.2/#files)

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install 'numpy==1.26.4' 'torch==2.7.1'
python prepare_mp20.py --download
python wbm.py
python check.py
```

The local commands use a separate replay-runs directory for the new outputs. Both preparers verify the source archive against the pinned release. `prepare_mp20.py` also accepts `--raw DIRECTORY`. `wbm.py` accepts `--source FILE.jsonl.gz`. Existing processed splits suffice for all commands below. No account key is needed to access these two releases. Every split stores string `ids`, ascending atomic numbers `z[rows,20]`, wrapped `f[rows,20,3]` and six log-metric coordinates `k[rows,6]`. Vacant positions carry atomic number and coordinates zero.

```sh
python experiment.py --data mp20-data --out replay-runs/mp20 --seeds 11 23 37 --device cpu --threads 2 --epochs 1
python evaluate.py --data mp20-data --out replay-runs/mp20 --seeds 11 23 37
python summarize.py replay-runs/mp20 --seeds 11 23 37
python audit.py --train mp20-data --data mp20-data --out replay-runs/mp20

python experiment.py --data wbm-data --out replay-runs/wbm --seeds 11 --device cpu --threads 1 --epochs 1
python evaluate.py --data wbm-data --out replay-runs/wbm --seeds 11
python summarize.py replay-runs/wbm --seeds 11
python audit.py --train wbm-data --data wbm-data --out replay-runs/wbm
```

For a direct transfer test, reuse MP-20 checkpoints. Retrieval and the score standardization then use only MP-20 training data. In the 128-case WBM test cohort, 18 reduced formulas and 9 complete cell compositions occur in the MP-20 training split. The two datasets have separate material ID namespaces, so the cross-dataset result allows shared chemistry and prototypes.

```sh
python experiment.py --data wbm-data --out replay-runs/mp20-to-wbm --replay replay-runs/mp20 --seeds 11 23 37 --device cpu
python evaluate.py --data wbm-data --fit-data mp20-data --out replay-runs/mp20-to-wbm --seeds 11 23 37
python summarize.py replay-runs/mp20-to-wbm --seeds 11 23 37
python audit.py --train mp20-data --data wbm-data --out replay-runs/mp20-to-wbm
```

One epoch ensures every training row appears as a positive anchor. Each update uses 16 successive shuffled rows. Three additional observed structures per anchor are sampled with replacement from the same exact cell composition. Both fitted arms receive the same 64 positive structures in the same order, use AdamW at 0.001, and clip gradients to norm 10. Cell features are standardized using training cells alone. `*-fit.json` records distinct training rows, optimizer updates, elapsed time and seed. `*.pt` checkpoints contain model state and training statistics. `--replay` generates with a checkpoint; `--updates` sets an explicit smaller or larger budget. For the main runs, 1,696 MP-20 or 12,819 WBM updates cover the training split once.

The evaluator takes the first 128 material IDs after sorting each held-out split by SHA256, a rule independent of the structures. For each it generates five predictions and compares with a training-geometry retrieval control. Exact compositions get their observed training structures. An unseen composition gets the nearest sorted atomic-number sequence with the same ordered multiplicities when available, otherwise the same atom count. Predicted cells retain all target species and counts. Its historical host positions are reassigned to the target's sorted species slots. No generated candidates undergo relaxation or validity filtering before scoring.

The energy score uses log volume per site and the minimum, median and 90th percentile of per-site periodic nearest-neighbor distances in Å. Features are scaled by the sample standard deviations across 2,048 predetermined training IDs. Periodic distances include nonzero self images and use a reciprocal-metric certificate for omitted translations. `check.py` tests this implementation against an explicit image enumeration, checks translation and atom-permutation symmetries, and computes gradients of both training objectives. `*.score.json` stores per-case energy and contact-screen outcomes. The unbiased estimate of the within-prediction distance term uses all ordered pairs of distinct draws. `geometry-summary.json` reports intervals from 10,000 paired resamples of the held-out IDs, conditional on the fitted seeds. Geometry retrieval can perform well here even when it uses an imperfect chemical template.

For pymatgen StructureMatcher, install the pinned matcher dependencies in a separate Python 3.10/3.11 environment. From that interpreter, for each completed run directory execute the corresponding command.

```sh
python -m pip install 'numpy==1.26.4' 'scipy==1.10.1' 'pymatgen==2024.4.13' 'spglib==2.5.0'
python match.py --run replay-runs/mp20 --data mp20-data
python match.py --run replay-runs/wbm --data wbm-data
python match.py --run replay-runs/mp20-to-wbm --data wbm-data
```

## Measured test results

Each cell in the table gives the conditional energy score followed by five-draw StructureMatcher recovery. A lower energy score means a closer distribution in the four measured features. The native WBM and MP-20→WBM runs use different training feature scales, so their absolute energy values measure different standardized distances. Each column describes 128 predetermined targets with five draws per target, averaged across the fitted seeds.

| Model | MP-20, three seeds | WBM, one seed | MP-20→WBM, three seeds |
|---|---:|---:|---:|
| Drift, one network call | 2.896 / 19.01% | 3.612 / 30.47% | 2.979 / 26.30% |
| Flow, one call | 3.163 / 14.84% | 4.012 / 21.88% | 3.331 / 21.09% |
| Flow, sixteen calls | 2.846 / 21.88% | 3.368 / 36.72% | 2.900 / 26.30% |
| Training geometry retrieval | 0.541 / 68.49% | 0.292 / 88.28% | 0.787 / 54.69% |

Across these three comparisons one-call drift improved the energy score over one-call flow. Sixteen-call flow achieved a lower score in all three. Relative to sixteen-call flow, the 95% paired intervals for drift's five-draw recovery difference included zero on both native datasets. Retrieval achieved much higher recovery even on WBM's formula-disjoint native split. The contact screen passed 88.44% of drift's raw MP-20 predictions and 94.22% on WBM. StructureMatcher tests agreement with a reference and the contact metric tests a distance threshold. Neither establishes thermodynamic stability. The primary score and matcher artifacts list the entire evaluation cohort and all usable outcomes. [MP-20 scores](runs/mp20/geometry-summary.json) · [WBM scores](runs/wbm/geometry-summary.json) · [Transfer scores](runs/mp20-to-wbm/geometry-summary.json) · [MP-20 matches](runs/mp20/structure-recovery-test.json) · [WBM matches](runs/wbm/structure-recovery-test.json)

Training on the 27,136 MP-20 records took 98.8–104.3 seconds per drift fit and 32.7–45.6 seconds per flow fit on two CPU threads. The full positive-anchor pass over 205,091 WBM training structures took 575.2 seconds for drift and 228.0 seconds for flow on one CPU thread. Each fitted arm used the same data and update budget within its dataset.

For timing replay, export only the learned weights and normalization arrays to portable NumPy first. `latency.py` measures from supplied atomic numbers to host arrays for the cell vectors and both Cartesian and fractional positions. It includes Gaussian/Haar sampling, condition transfer, network evaluations, periodic wrapping, device synchronization, host transfer, symmetric-matrix exponential, and Cartesian site conversion. The warm retrieval control runs on CPU, including its per-request donor lookup and cell decoding. Model loading, initial donor-table construction, and writing files lie outside the timed intervals; the metadata records donor setup separately. For each fitted seed, dataset and device there are 15 interleaved observations after three warmups at each batch size 1, 64 and 256.

```sh
python latency.py --data mp20-data --run replay-runs/mp20 --export --device cpu --seeds 11 23 37
python latency.py --data wbm-data --run replay-runs/wbm --export --device cpu --seeds 11
python latency.py --data mp20-data --run replay-runs/mp20 --device cuda --seeds 11 23 37
python latency.py --data wbm-data --run replay-runs/wbm --device cuda --seeds 11
```

The table gives single-crystal latency and complete batch-of-256 latency in milliseconds. Each entry is the median of the fitted-seed medians. CPU used one thread. The GPU measurements used a local NVIDIA GeForce RTX 3060 Laptop GPU with WSL Linux, Python 3.9.25, PyTorch 1.9.1 and CUDA 11.1. The same portable parameters, host decode and batch sizes were used on both platforms. Operating system, PyTorch version, host load and CPU scheduling can affect these observations. The WBM run used one fit and the MP-20 runs used three. CPU retrieval times varied particularly under WSL, and all individual observations are retained.

| Dataset and hardware | One-call drift, 1 / 256 | One-call flow, 1 / 256 | Sixteen-call flow, 1 / 256 |
|---|---:|---:|---:|
| MP-20, CPU | 1.18 / 15.14 | 1.30 / 15.66 | 10.81 / 217.90 |
| MP-20, RTX 3060 | 3.55 / 3.50 | 3.56 / 4.42 | 34.80 / 38.32 |
| WBM, CPU | 2.54 / 34.10 | 2.69 / 32.28 | 29.85 / 467.38 |
| WBM, RTX 3060 | 3.49 / 4.33 | 3.60 / 4.96 | 31.71 / 38.46 |

The two neural one-call methods take similar time on these small cells. Batched sixteen-call flow takes 38.3–38.5 ms on the GPU. The CPU retrieval control has batch-of-256 medians of 1.42 ms on the MP-20 run and 3.27 ms on the WBM run. [MP-20 CPU observations](runs/mp20/latency-cpu.csv) · [MP-20 CUDA observations](runs/mp20/latency-cuda.csv) · [WBM CPU observations](runs/wbm/latency-cpu.csv) · [WBM CUDA observations](runs/wbm/latency-cuda.csv)

## Jean Zay

Stage the `scaled` directory with the six processed NPZ files and split manifests on shared storage before submission. From that directory, replace `PROJECT` with your H100 allocation. Both jobs fit three independent seeds on CUDA and compute geometric scores locally on the allocated node. The result directory also contains the job description, hardware, loaded modules, input hashes and exact training sources. Run `match.py` on the exported NPZs from the complete research tree in the matcher environment above. The script imports `standard_match.py` and `data.py` from its parent directory. The Slurm allocation runs training and geometric evaluation with the staged `scaled/` directory.

```sh
sbatch --account=PROJECT@h100 jean_zay_scaled.slurm mp20
sbatch --account=PROJECT@h100 jean_zay_scaled.slurm wbm
```

The default output root is `job-results/<job-id>`; export `CRYSTAL_RUN_ROOT` to use another shared directory. The script uses the H100 software environment documented by IDRIS; `CRYSTAL_TORCH_MODULE` can select another locally installed GPU PyTorch release. The script passed local Bash syntax checking. The session had no configured Jean Zay connection or allocation for remote submission. [IDRIS H100 software example](https://github.com/idriscnrs/tutorials_jean-zay_picotron)
