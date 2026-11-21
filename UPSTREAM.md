# External inputs and attribution

Run the commands below in the study directory containing `data.py`, `scaled/` and `lattice_symmetry/`. The periodic generation experiments and the capacity-controlled comparisons are documented in `lattice_symmetry/v2/` and `lattice_symmetry/v3/`. Each comparator has its own input requirements.

## MP-20

CrystalDrift uses the published MP-20 split of Materials Project structures. The WyckoffDiff release supplies the three compressed CSVs used here. The dataset attribution in `data/README.md` identifies Materials Project under CC BY 4.0. The transformation to padded fractional coordinates, atomic numbers and log lattice metrics is described there and in `scaled/prepare_mp20.py`. Original release hashes are recorded in `data/manifest.json`. The split sizes are 27,136 training, 9,047 validation and 9,046 test structures. [MP-20](https://github.com/txie-93/cdvae/tree/main/data/mp_20) · [WyckoffDiff data provenance](https://github.com/httk/wyckoffdiff/blob/main/data/README.md) · [Materials Project](https://materialsproject.org/)

```sh
python scaled/prepare_mp20.py --download --raw data
python data.py
```

The first command verifies the upstream SHA-256 checksums and creates the complete split arrays under `scaled/mp20-data/`. The second creates the earlier binary subsets under `data/`.

## External structure datasets

| Experiment | Rebuild command | Source |
|---|---|---|
| WBM formula-disjoint comparison | `python scaled/wbm.py` | [Matbench Discovery relaxed WBM release](https://doi.org/10.6084/m9.figshare.22715158), file 53161832. The loader checks MD5 before conversion. |
| Alexandria transfer | `python scaled/prepare_alex.py --download` after preparing MP-20 | [MatterGen Alex-MP-20](https://github.com/microsoft/mattergen/tree/92423660a8bd70e83679086e88f88596d484dc16/data-release/alex-mp). The loader pins the commit and verifies the complete ZIP by SHA-256. |

## Wyckoff tables

The conventional-cell Wyckoff charts in `lattice_symmetry/v2/wyckoff/charts.npz` are derived from PyXtal 1.0.9. The project records the wheel and source table hashes in `chart-source.json`. The PyXtal copyright and MIT license are in `PyXtal-LICENSE.txt`. To regenerate the chart in an environment with NumPy and pymatgen, run the following commands. [PyXtal 1.0.9](https://pypi.org/project/pyxtal/1.0.9/)

```sh
python -m pip download --only-binary=:all: --no-deps pyxtal==1.0.9 -d upstream
python lattice_symmetry/v2/wyckoff/extract_charts.py upstream/pyxtal-1.0.9-py3-none-any.whl
```

## Model comparisons and optional physical checks

| Component | Source and reproduction |
|---|---|
| uFlowCSP Cartesian MP-20 model | Sourin Dey and Jianjun Hu released the Cartesian implementation under MIT and the checkpoint under CC BY 4.0. CrystalDrift's native replay reads snapshots of the pinned upstream source at runtime. Retain the `.py.txt` files in `uflowcsp-results/` and `lattice_symmetry/v3/uflow/` with `uflowcsp-results/cartesian-LICENSE`. Run `python uflowcsp_weights.py` to recover the EMA tensors. Set `UFLOW_WEIGHTS` to the resulting `ema.safetensors` inside the system temporary directory's `codex-uflowcsp-69274087/` folder. The environment and order of the subsequent commands are in `revision/full/external/README.md` and `lattice_symmetry/v3/README.md`. [Source at the replayed commit](https://github.com/usccolumbia/uFlowCSP/tree/e87af8483e6d45b0217b4ede430862e992d2b1d0) · [Checkpoint](https://doi.org/10.6084/m9.figshare.33974422.v1) |
| DiffCSP++ | `lattice_symmetry/v2/wyckoff/external/source.json` identifies the upstream commit, weight archive and individual file hashes. The source uses the MIT license preserved in `external/LICENSE`. To rebuild `external/weights.npz`, clone upstream into `upstream/DiffCSP-PP`, check out commit `e82e7ffa7cb2a383bde69b067a343ca137d73e47`, run `python lattice_symmetry/v2/wyckoff/fetch_diffpp.py upstream/diffpp.ckpt` and then `python lattice_symmetry/v2/wyckoff/prepare_diffpp.py upstream/diffpp.ckpt upstream/DiffCSP-PP`. [Pinned DiffCSP++ source](https://github.com/jiaor17/DiffCSP-PP/tree/e82e7ffa7cb2a383bde69b067a343ca137d73e47) |
| CrystalFlow | `python expansion/external/fetch_source.py upstream/crystalflow/source` verifies the pinned Git blobs. `python expansion/external/fetch_joint.py upstream/crystalflow` recovers the joint MP-20 checkpoint. `expansion/external/README.md` gives the conditional-weight extraction, reference pool and evaluation commands. [Code](https://github.com/ixsluo/CrystalFlow/tree/9c25ff0245d787efd87c9d1d797a8968443608cc) · [Checkpoint release](https://github.com/ixsluo/CrystalFlow/releases/tag/v1.0.0-alpha.1) · [Published reference structures](https://doi.org/10.6084/m9.figshare.29978608.v3) |
| DiffCSP property evaluator | `python expansion/denovo/fetch_predictor.py` fetches the pinned MP-20 property checkpoint, scalers, configuration, MIT license, and reference implementation. The evaluator verifies the files against upstream Git hashes. [Pinned DiffCSP release](https://github.com/jiaor17/DiffCSP/tree/7121d159826efa2ba9500bf299250d96da37f146) |
| Historical DFT comparison | The four Mg–Ga configurations use Quantum ESPRESSO and two PSLibrary pseudopotentials. Place both files in `expansion/dft/pseudos/`. The filenames and checksums are recorded in `expansion/dft/setup.sh`, with the execution environment in `expansion/dft/README.md`. [Mg PAW](https://pseudopotentials.quantum-espresso.org/upf_files/Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF) · [Ga PAW](https://pseudopotentials.quantum-espresso.org/upf_files/Ga.pbe-dn-kjpaw_psl.1.0.0.UPF) |

The structural matcher uses pymatgen 2024.4.13 and spglib 2.5.0. The fast environment-fingerprint routines preserve pymatgen's license in `expansion/denovo/pymatgen-LICENSE`. Versions for the optional physical-screening and native-model environments are listed in `lattice_symmetry/v3/README.md`.
