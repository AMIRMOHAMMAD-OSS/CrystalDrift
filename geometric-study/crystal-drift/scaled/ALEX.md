# Alexandria validation transfer and the Alex-MP-20 release

MatterGen's pinned Alex-MP-20 archive contains **607,683 training structures and 67,521 validation structures**. The third CSV provides formation-energy, space-group and other metadata for 968,035 reference IDs. Only the first two CSVs contain structures. The release combines Alexandria and Materials Project records. Its published training criteria include at most twenty atoms and a formation energy within 0.1 eV/atom of the convex hull. Cite the constituent materials databases when reusing the data. [Pinned release and attribution](https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/data-release/alex-mp/README.md) · [Archive](https://media.githubusercontent.com/media/microsoft/mattergen/92423660a8bd70e83679086e88f88596d484dc16/data-release/alex-mp/alex_mp_20.zip)

## Frozen MP-20 transfer

The source models used the 27,136 MP-20 training structures. Of the 67,521 Alex-MP-20 validation records, 2,252 have a literal MP-20 training ID and 5,224 have a reduced formula represented in that training set. We restricted the evaluation to literal Alexandria `alex<…>` IDs, excluded source training IDs and embedded MP/MVC aliases, and excluded every known training formula. This left 60,672 eligible records. The first 128 in ascending SHA-256 order of their UTF-8 IDs formed the cohort. The filter concerns elemental identities and reduced stoichiometry. Common anonymous prototypes can still occur in both datasets.

Each frozen MP-20 checkpoint (seeds 11, 23 and 37) supplied five unfiltered proposals per target. Retrieval selected MP-20 training geometries using the target composition, with the same five-draw budget. We averaged each target's outcome across seeds before resampling the 128 targets 10,000 times. The intervals are conditional on the three fitted checkpoints. `match.py` used pymatgen StructureMatcher with relative lattice tolerance 0.3, site tolerance 0.5, angle tolerance 10°, primitive reduction and volume rescaling. Contact validity requires every atom's nearest periodic neighbor to lie at least 0.5 Å away. The fair energy score measures log volume per atom and three periodic nearest-neighbor distance statistics. Smaller values indicate better conditional geometry.

| MP-20 source method | Match, 1 draw | Match, 5 draws | Fair geometry energy | Contact-valid draws |
|---|---:|---:|---:|---:|
| CrytalDrift | 4.95% | 14.84% | 3.459 | 93.65% |
| Flow, 1 step | 4.69% | 9.64% | 3.720 | 90.16% |
| Flow, 16 steps | 6.25% | 13.54% | 3.374 | 93.70% |
| MP-20 geometry retrieval | 28.13% | 28.13% | 1.337 | 100% |

At five draws, CrytalDrift exceeded one-step flow by 5.21 percentage points (paired 95% interval 1.56 to 8.85). Its difference from 16-step flow was 1.30 points (−1.30 to 3.91), and its difference from retrieval was −13.28 points (−21.35 to −5.21). All 7,680 candidate–target outcomes were scored without a matcher exception. [Cohort IDs and eligibility counts](alex-validation-transfer/cohort.json) · [Recovery and paired intervals](runs/mp20-to-alex/structure-recovery-val.json) · [Geometry and paired intervals](runs/mp20-to-alex/geometry-summary.json)

## Reproduce the conversion and transfer

Run the commands from `crystal-drift/scaled`. `prepare_alex.py` checks the 186,506,244-byte archive against SHA-256 `6688e652e45fac4ac8b016dbc4a3280afde9ba687bdacfd8c676775ed19b6228`. It also checks output hashes, occupancies, formula sums, positive Gram eigenvalues, finite coordinates, record counts and unique IDs across both splits. The verified local archive is `alex-data/alex-verified.zip`. With an existing archive, omit `--download`. The converter and matcher were run with Python 3.10/3.11-compatible dependencies. The replay additionally needs the PyTorch environment and frozen `runs/mp20/` checkpoints described in the common run guide. [Common environment and MP-20 preparation](RUN.md)

```sh
python -m pip install 'numpy==1.26.4' 'scipy==1.10.1' 'pymatgen==2024.4.13' 'spglib==2.5.0'
python prepare_mp20.py --download
python prepare_alex.py --download
python alex_transfer.py
python match.py --run runs/mp20-to-alex --data alex-validation-transfer --workers 4
```

The importer retains the source CIF cell basis for 675,199 structures and sorts occupied sites by atomic number. Five training CIFs contain more than twenty deposited positions. Their primitive structures have index three, one with 60→20 sites and four with 27→9. The importer reconstructs the integer supercell and verifies its lattice and site equivalence without volume rescaling. No source structures were lost. The CIF site count differs from the CSV `num_sites` field for 334 training and 43 validation records. The report preserves discrepancy counts and the five reduction IDs. The publisher's own cache pipeline applies primitive and Niggli reduction more broadly. These prepared NPZ files use the same chart convention as the MP-20 transfer source wherever the site cap permits it. [Release conversion audit](alex-data/manifest.json) · [MatterGen cache implementation](https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/data/dataset.py#L306-L343)

## Optional Alex-MP-20 training

For a fresh Alex fit, `--training-view` builds a separate directory accepted by the shared trainer. A fixed SHA-256 rule assigns entire reduced formulas from the released training data to internal tuning. The tuning partition contains 12,029 structures across 9,761 formulas. The fit partition contains 595,654 structures across 481,751 other formulas. The common trainer reads a third file called `test.npz`, a byte-for-byte copy of the entire **released validation** partition. Of its 67,521 rows, 18,794 share a reduced formula with the derived fit partition and 361 share a formula with internal tuning. The frozen transfer results above use only the separately filtered 128 validation IDs and the original MP-20 models. [Training-view manifest](alex-training-view-data/manifest.json)

```sh
python prepare_alex.py --download --skip-transfer-cohort --training-view alex-training-view-data
```

The standard runner accepts CPU, CUDA or Apple MPS through `--device`. This example starts a local MPS fit for seed 11 in the PyTorch environment.

```sh
python experiment.py --data alex-training-view-data --out runs/alex-local --device mps --seeds 11
```

On Jean Zay, stage the CPU-prepared `alex-training-view-data/` beside the scripts and submit from `scaled/` with your own project allocation. The batch script runs seeds 11, 23 and 37 and records scheduler state, software versions, data hashes and checkpoints. The reported experiment used local frozen-checkpoint inference. This cluster submission is an optional training procedure.

```sh
sbatch --account=PROJECT@h100 jean_zay_scaled.slurm alex-training-view
```
