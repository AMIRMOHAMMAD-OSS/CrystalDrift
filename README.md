# CrystalDrift

Project repository: [github.com/AMIRMOHAMMAD-OSS/CrystalDrift](https://github.com/AMIRMOHAMMAD-OSS/CrystalDrift).

Code, experiments and manuscript for *CrytalDrift — One-shot generation of crystal structures*, by AmirM. MohammadHosseini. The paper studies a training-time drifting field in periodic feature space and a one-evaluation generator that updates occupied Wyckoff orbits and the lattice. The repository name uses CrystalDrift. Several manuscript and experiment filenames retain the original spelling, CrytalDrift.

[Manuscript PDF](CrytalDrift_Manuscript/CrytalDrift.pdf) · [LaTeX source](CrytalDrift_Manuscript/main.tex) · [arXiv source archive](CrytalDrift_Manuscript/CrytalDrift_arXiv_Source.zip)

## Run the symmetry checks

Use Python 3.10 with PyTorch, NumPy and SciPy and run the following commands from the repository root. The first command changes to the working directory expected by the experiment code. The output files written by the two probes contain the numerical residuals.

```sh
cd geometric-study/crystal-drift
python -m unittest lattice_symmetry.test_symmetry lattice_symmetry.test_spacegroup_seed
python -m lattice_symmetry.v3.geometry.verify --device cpu --output verification-local.json
python -m lattice_symmetry.v3.geometry.verify_decoder --device cpu --output decoder-local.json
```

The first command inside the study directory checks changes of basis, origin, atom ordering and Cartesian frame on small deterministic fixtures. The next two exercise the descriptors, the finite-bank drifting field and the orbit decoder used in the capacity comparison. There is no dataset download or checkpoint load in these tests. See the [geometry checks](geometric-study/crystal-drift/lattice_symmetry/v3/geometry/README.md) for the meaning of each residual.

## Source and experiments

From the repository root, run `python verify_published_scores.py` to recompute target counts, structural recovery and matched displacement for the 27 rows of the main comparison and the 162 rows of the seed table. This reads 63 bundled per-target score arrays. The timing columns and fitted checkpoints have separate provenance in the experiment records below.

| Task | Entry point |
| --- | --- |
| Inspect the periodic, contact and reciprocal descriptors and the training field | [descriptors.py](geometric-study/crystal-drift/lattice_symmetry/v3/geometry/descriptors.py) |
| Construct the Wyckoff seed, prepare chart coordinates and apply the one-step decoder | [sampler.py](geometric-study/crystal-drift/lattice_symmetry/v3/models/sampler.py) · [network.py](geometric-study/crystal-drift/lattice_symmetry/v3/models/network.py) |
| Train orbit generators and the same-backbone MeanFlow control | [training instructions](geometric-study/crystal-drift/lattice_symmetry/v3/models/README.md) · [train.py](geometric-study/crystal-drift/lattice_symmetry/v3/models/train.py) |
| Reproduce split construction, training exposure and target-wise reconstruction | [analysis plan](geometric-study/crystal-drift/lattice_symmetry/v3/protocol/analysis-plan.md) · [experiment commands](geometric-study/crystal-drift/lattice_symmetry/v3/README.md) |
| Inspect reported recovery, capacity and timing comparisons | [MAIN-COMPARISON.csv](geometric-study/crystal-drift/lattice_symmetry/v3/release/MAIN-COMPARISON.csv) · [SEED-RESULTS.csv](geometric-study/crystal-drift/lattice_symmetry/v3/release/SEED-RESULTS.csv) · [claim-to-evidence crosswalk](geometric-study/crystal-drift/lattice_symmetry/v3/release/CLAIM-AUDIT.md) |
| Compare earlier site-based and stochastic periodic models | [earlier implementation](geometric-study/crystal-drift/README.md) · [periodic model experiments](geometric-study/crystal-drift/lattice_symmetry/v2/) |
| Inspect ordered de novo pools and run physical screening | [de novo analysis](geometric-study/crystal-drift/lattice_symmetry/v3/denovo/) · [screening and earlier experiments](geometric-study/crystal-drift/expansion/README.md) |
| Edit the paper, figures and Wyckoff design files | [current manuscript](CrytalDrift_Manuscript/) · [figure design](CrystalDrift_Figure1_Design/) |

The earlier experimental READMEs describe their historical manuscripts. Earlier theory and review files are in [CrystalDrift_Theory_Draft](CrystalDrift_Theory_Draft/) and [output/crystal-review](output/crystal-review/). Historical launchers record the original workstation paths. Use the Python commands in the experiment READMEs or set the paths for your environment. The version in `CrytalDrift_Manuscript/` is the current paper. Rebuilding it requires two pdfLaTeX passes from that directory.

```sh
cd CrytalDrift_Manuscript
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Data, environments and replay

The core study uses the official MP-20 train, validation and test partitions. The WBM and Alexandria experiments have separate data provenance. See [UPSTREAM.md](UPSTREAM.md) for acquisition instructions and third-party notices. The [study environment](geometric-study/crystal-drift/lattice_symmetry/v3/README.md) records the training, matcher and relaxation package versions. Python environments and hardware differ across these experiments. PyTorch 2.6.0 with CUDA 12.4 was used for the capacity-controlled GPU study. The pinned matching environment used Python 3.10, NumPy 1.26.4, SciPy 1.10.1, Numba 0.59.1, pymatgen 2024.4.13 and spglib 2.5.0.

The Git tree contains the source, manuscript, input manifests, summary tables, regression fixtures, a binary-model demonstration checkpoint and selected processed arrays. Larger project-produced arrays, checkpoints, prediction shards and images are organized into separately distributable ZIP volumes. [Volume checksums](release-data/archives.sha256) identify their exact bytes. The [asset file index](release-data/asset-files.tsv) maps each recorded output to a volume and gives its per-file checksum. Extract available volumes at the repository root to restore these paths. For example, after placing the volumes next to each other:

```sh
python -m zipfile -t /path/to/CrystalDrift-results-01.zip
python -m zipfile -e /path/to/CrystalDrift-results-01.zip .
```

The authors' [historical full-run manifest](geometric-study/crystal-drift/lattice_symmetry/v3/release/EVIDENCE-MANIFEST.json) describes a separate experiment snapshot. Results recorded only on remote training or backup volumes are not regenerated by extracting the Git checkout. Full replay requires its declared matching inputs, weights and ordered predictions. Upstream datasets and large third-party model weights have their own provenance and distribution terms.
