# Independent energy and force check on 32 generated crystals

We evaluated draws 1–32 of the fixed 1,024-proposal stream. All 32 received an initial evaluation and up to 100 FIRE steps that could move both atoms and lattice vectors at zero applied pressure. No candidate was rejected before this evaluation. The underlying calculator was the unmodified official CHGNet MPtrj checkpoint 0.3.0 from package 0.3.8. The release wheel SHA-256 is 143ce903ce4d4650a24c7855b696dadb1238d414ca5778c44f06e8a736d01006. The checkpoint SHA-256 is d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1. [Official checkpoint documentation](https://chgnet.lbl.gov/#available-pretrained-models) · [CHGNet paper](https://doi.org/10.1038/s42256-023-00716-3)

## Results

| Measure | Generated proposals | Random held-out Materials Project controls |
|---|---:|---:|
| Sample count | 32 | 12 |
| Reached ASE cell-plus-position stopping rule | 23 | 12 |
| Final maximum physical atomic force ≤0.1 eV/Å | 17 | 12 |
| Median maximum force, initial → final (eV/Å) | 32.44 → 0.097 | <0.001 → <0.001 |
| Median nearest periodic contact, initial → final (Å) | 1.440 → 2.503 | 3.065 → 3.053 |
| Contacts below 0.5 Å, initial → final | 1 → 3 | 0 → 0 |
| Final shortest contact below 60% of corresponding summed covalent radii | 7 | 0 |
| Median final minus initial energy (eV/atom) | −4.416 | 0.000 |
| Median internal atomic displacement (Å) | 0.723 | 0.000 |
| Median optimizer wall time per structure (seconds) | 3.118 | 0.006 |

FIRE uses the forces of ASE’s FrechetCellFilter to decide convergence. That combined coordinate system contains deformation variables. The physical atom-force threshold is reported separately. Nearest contacts include periodic images of the same atom. The radii criterion is a geometric diagnostic whose interpretation depends on the bonding environment. Relaxation lowered every generated energy, yet three final structures had overlaps under 0.5 Å. For Mn₆Ir₂ (draw-0022), the nearest distance shrank from 0.650 to 0.212 Å and the final maximum force was 3.55 eV/Å at the step cap. This illustrates extrapolation outside the potential’s reliable region.

## Matched energy comparisons

The random controls were 12 draws without replacement from the 1,229 multisite test cells, using NumPy seed 20261120. For every generated reduced formula, we also searched the full 45,229-record training, validation and test release. We chose one reference before scoring: prefer the official test split, then validation, then training; within a split choose fewer sites, then lexicographically smallest material ID. There were 23 matches, all with distinct reference IDs. Seven were from test, three from validation and 13 from training. All 23 reference structures converged.

Twenty of the 23 generated candidates finished above their selected same-formula reference on the CHGNet energy scale. The median signed difference was +0.041 eV/atom; values ranged from −0.103 to +9.967. Negative values can reflect a different local structure or potential error. The reference selection does not search for the lowest-energy known phase. Nine generated compositions lacked a reference in this release. CHGNet itself was pretrained with Materials Project trajectories, so the real controls are held out from the generator and may overlap the potential’s training data.

All 44 elements in the candidates also had at least one elemental reference in the release. For each element we chose the lowest deposited energy above hull, then the lowest deposited formation energy, then fewer sites and finally material ID. Those deposited quantities select reference structures only. We recomputed their energies with the same calculator and optimizer as the candidates. All 44 elemental cells converged. For 13 of 32 candidates the final CHGNet energy exceeds the concentration-weighted combination of the two elemental endpoints. The remaining 19 fall below that combination. The median signed difference is −0.046 eV/atom. The endpoints omit competing compounds at other binary compositions, so relative-to-element energies cannot establish hull stability, synthesizability or dynamical stability. [Elemental comparison](../elemental-screen/candidate-gaps.csv) · [Selected endpoints](../elemental-screen/selection.json)

## Reproduction and outputs

The generator’s candidate archive has SHA-256 b01ccf21e60da1a2919bc5e4baf2c146047f6eac97dc8088420eb77d885dba56. The protocol used CPU PyTorch 2.7.1, ASE 3.23.0, pymatgen 2024.4.13, one PyTorch thread per worker and four process workers. Optimizer wall times measure the separate refinement stage. They exclude initial model loading, the initial single-point evaluation, file I/O and generator inference. The main generator/control/reference run took 49.0 s wall time after selection, including worker startup. One control was reused from the prior smoke check. Machine load can affect these timings.

Run the following commands from the crystal-drift project directory after installing the pinned packages.

~~~shell
python relax_candidates.py
python elemental_screen.py
~~~

The per-structure records include potential warnings, force and stress diagnostics, initial and final cells, status and wall time. Internal displacement compares identical atom indices in the final cell using periodic minimum-image distances and removes the shared translation. The strain diagnostic is the largest absolute logarithm of a principal stretch. Both retain information that a scalar energy change discards. [Per-sample measurements](results.csv) · [Cohort definitions](selection.json) · [Runtime manifest](run.json) · [Initial and relaxed structures](cifs/) · [Generated relaxed coordinates](relaxed-candidates.npz)
