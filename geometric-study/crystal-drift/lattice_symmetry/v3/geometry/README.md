# Geometry, operations, and independent checks

The manuscript supplement is `supplement.tex`. These checks use the production feature, finite-bank and sampler entry points. The NumPy oracles enumerate integer images independently. Property fixtures are generated deterministically from the recorded seeds in `verify.py` and `verify_decoder.py`. Calibration and prototype analyses read MP-20 material IDs from the registered split.

| Mathematical or empirical statement | Implementation | Numerical evidence |
|---|---|---|
| Physical image shell, center-wise harmonic contraction and species slots | `../../v2/neighbors.py`, `../../v2/physical.py`, `descriptors.feature_blocks` | `verification-{cpu,cuda}.json`; nine independently re-expressed queries or references at two precisions |
| Inter-orbit histograms and second moment over typed unordered pairs | `descriptors.orbit_layout`, `inter_orbit_contacts` | Independent explicit pair and image loops, relabelled repeated orbits, finite-difference derivatives and variance-blind witness in `verify.py` |
| Complete physical reciprocal sphere and phase cancellation | `descriptors.reciprocal_indices`, `reciprocal_shells` | Independent complex-amplitude oracle, all-index sphere vs failing fixed integer box, directional derivatives and homometric/long-range fixtures in `verify.py` |
| Finite-field representation invariance, sign swap and stopped parameter gradient | `descriptors.bank_direction`, `drift_loss` | Twelve arm/affinity cells per precision/device in `verification-{cpu,cuda}.json`; Laplace parity with `scaled.family.within_direction`; fixed-anchor counterexample in `train-bank-audit.json` |
| Training-only normalizers, temperature calibration and signal availability | `models/prepare.calibrate`, `descriptors.fit_stats` | `models/cache/calibration.npz`, `bank_audit.py`, `train-bank-audit.json`; 256 distinct retained training anchors, per-block responses to 5% dilation |
| Equivariant orbit-token update, positive cell metric, preserved chart and requested stabilizer | `models/sampler.compose`, `models/network.pack`, `OrbitNet.forward` | `verify_decoder.py`, `decoder-*.json`; nine space groups including screw, glide, centering, repeated occupancies and seven zero-degree orbits. Gauge and representation transforms perturb every bank independently |
| Conventional-to-primitive orbit reduction | `models/sampler.orbit_chart`, `group_ops` | `verify_charts.py`, `primitive-charts.json`; 1,731 charts, two parameter choices per chart, expected orbit cardinalities and native fractional sets compared bidirectionally to group action and unreduced conventional images |
| Average-velocity objective under fixed endpoint assignment | `models/prepare.align_endpoint`, `models/train.meanflow` | `verify_meanflow.py`, `meanflow-objective-test.json`; independent centered differentiation of the actual loss and parameter gradient for six training seeds |
| Richer radial resolution with common harmonic orders | `descriptors.dense_pair_shells` | `radial_check.py`, `radial4-test.json`; 112 additional channels agree with the released four-shell pair map, a direct NumPy enumeration and independent query/reference transforms |
| Template reachability, retained groups and anonymous geometric analogues | `models/sampler.Sampler.shortlist`, `letter_tables.py`, `reachability.py`, `shared_geometry.py`, `subgroups.py` | Per-target Boolean matrices and per-donor/panel records in `audits/`. Coverage and matcher truth are consulted retrospectively |

## Replay

`verification-cpu.json` also compares identical oxygen sites under two admissible subgroup partitions. Splitting the inversion pair into two P1 orbits changes the contact block by 1.432 in Euclidean norm. The angular, variance and reciprocal blocks agree exactly in this example at both precisions.

Run from the crystal-drift root with Torch, NumPy and one CPU thread. Cell lengths are in angstroms. Set `--device cuda` on the corresponding GPU checks for the additional backend. The release used FP32 training. The NumPy oracles and finite differences also run in FP64.

~~~sh
python -m lattice_symmetry.v3.geometry.verify
python -m lattice_symmetry.v3.geometry.verify --device cuda
python -m lattice_symmetry.v3.geometry.verify_decoder --checkpoint lattice_symmetry/v3/models/fits/small-angular-laplace-11/full.pt --output lattice_symmetry/v3/geometry/decoder-small-angular-full.json
python -m lattice_symmetry.v3.geometry.bank_audit
python -m lattice_symmetry.v3.geometry.verify_meanflow
python -m lattice_symmetry.v3.geometry.radial_check
python -m lattice_symmetry.v3.geometry.native_ops
python -m lattice_symmetry.v3.geometry.verify_charts
python -m lattice_symmetry.v3.geometry.reachability dev confirm family shared test
~~~

The normalizer tables came from the chart release's pinned PyXtal 1.0.9 wheel. To reconstruct the compact table, pass that wheel to `python -m lattice_symmetry.v3.geometry.letter_tables`. Each of the 230 tables includes identity and closes under composition, with 878 distinct letter permutations in total. Apply one group-specific permutation simultaneously to all element-labelled occupations. A case can miss its exact occupancy and still match approximately under the public geometrical matcher.

The commands below use the existing pinned pymatgen 2024.4.13 and spglib 2.5.0 interpreter. Set `MATCHER_PYTHON` to its executable. The subgroup audit checks all primitive operation tables against an independent spglib Hall database before inspecting a single candidate. For raw samples the origin is fixed by the orbit parameterization. For relaxed samples it is transported by the mean wrapped fractional site displacement. We report group number, containment and equality of the finite fractional operation sets modulo integer lattice translations, with the same relaxation-panel denominator throughout.

~~~sh
"$MATCHER_PYTHON" -m lattice_symmetry.v3.geometry.shared_geometry
"$MATCHER_PYTHON" -m lattice_symmetry.v3.geometry.subgroups prior
~~~

Pass the additional completed pool names to `subgroups` to recheck their hash-selected candidates. Select any completed default symbolic-lookup method tags as additional arguments to `python -m lattice_symmetry.v3.geometry.conditional PANEL TAG...`. The method must use the registered lookup and draw order. For each protected panel, `python -m lattice_symmetry.v3.geometry.family_effects PANEL LABEL TAG...` computes learned-minus-prior recovery, carrying training seeds together in the family-cluster bootstrap. The shared-family comparison additionally stratifies on existence of a continuous structural analogue under the pinned matcher's anonymous element mapping. An exhaustive miss refers to the declared operational family and the protected training memory.

The actual field is a signed finite-bank displacement. Matching distributions gives zero expected field only under the stated conditional joint exchangeability. Stable structures, guaranteed recovery or a unique solution do not follow. The geometric features operate during training. Inference receives the sampled symbolic information and geometry specified by the corresponding pipeline.

The secondary validation search was registered in `ARRANGEMENT-PROTOCOL.md`. Run `python -m lattice_symmetry.v3.geometry.arrangements register` and then `"$MATCHER_PYTHON" -m lattice_symmetry.v3.geometry.arrangements classify`. The registration and state are written atomically under `audits`. Reconstruct the supplementary shortlist and vector plot with `python -m lattice_symmetry.v3.geometry.reachability arrangement arrangement-control` and `python -m lattice_symmetry.v3.geometry.plot_coverage`. The registered list excludes every ID identified in prior validation panels, the first v3 development/confirmation panels and the raw binary validation subsets previously used for model choice.

In the stand-alone v3 evidence extraction, `python -m lattice_symmetry.v3.geometry.arrangements replay-register` verifies the frozen exclusion list against the registration hash, reconstructs every eligible training and validation catalog and checks every registered target and ordered donor. Regenerating historical exclusion provenance with `register` also uses the archived earlier validation panels.

To replay the independent PyXtal check, run `python -m lattice_symmetry.v3.geometry.pyxtal_roundtrip make` using the model environment. With PyXtal 1.0.9 installed alongside the pinned matcher, run `"$MATCHER_PYTHON" -m lattice_symmetry.v3.geometry.pyxtal_roundtrip check`. The checks include in-basis requested operations from spglib, recognized Hall number, simultaneous setting-equivalent occupied pattern and a bidirectional native-to-conventional geometry round trip. For a stock matcher failure the records also report independent primitive reductions.

`python -m lattice_symmetry.v3.geometry.raw_templates PANEL TAG...` independently reconstructs the five coverage flags from the ordered prior's saved groups and complete element-labelled occupations, checks the updated models' metadata against that stream, and tests every nonempty symbolic and cell donor ID for training eligibility. `python -m lattice_symmetry.v3.geometry.ranking_coverage dev confirm family shared arrangement arrangement-control test` runs the frozen selector, the train-frequency selector and chemical ordering over the same formula-dependent shortlist and cross-checks saved generated ranks where present. The code checks equality of the twenty-draw coverage ceilings before writing its audit.

The CIF catalog command is `"$MATCHER_PYTHON" -m lattice_symmetry.v3.denovo.cif_catalog ARM... --output OUTPUT_DIRECTORY`. Each archive contains one explicitly expanded P1 CIF per successful raw geometry and an ordered CSV describing every attempt, including failed geometry and independent spglib detection. The metadata under `denovo/catalog` records both SHA-256 digests, parser-verified metric/site round trips and per-group counts for each arm. Use `"$MATCHER_PYTHON" -m lattice_symmetry.v3.denovo.group_distribution ARM...` to compare the raw distributions with the public test structures and the exact training structures that supplied the formulas for these draws.

The commands below use the project root as working directory.

```sh
python -m lattice_symmetry.v3.geometry.decorated_witness
python -m lattice_symmetry.v3.geometry.collision_sweep
python -m lattice_symmetry.v3.geometry.plot_collision
python -m lattice_symmetry.v3.geometry.crystal_gallery
python -m lattice_symmetry.v3.geometry.crystal_gallery --arm contacts-medium-long --stem crystal-gallery-extended
python -m lattice_symmetry.v3.denovo.plot_symmetry
python -m lattice_symmetry.v3.geometry.ranking_effects
python -m lattice_symmetry.v3.geometry.selector_outcomes
python -m lattice_symmetry.v3.geometry.volume_strata
python -m lattice_symmetry.v3.geometry.write_coverage_section
```

Run `python -m lattice_symmetry.v3.geometry.official_diagnostics` for the public split after scoring. It checks the actual ordered template streams against the pattern-coverage flags and updates the conditional, volume, selector and typeset results for every available lookup refiner. With the later raw shards on the WSL scratch disk, set `CRYTALDRIFT_RAW` to that directory. The final seven CIF pools can be reproduced by passing `prior angular-small contacts-small angular-medium-long contacts-medium-long released-angular released-uflow` to the CIF and distribution commands, in that order.

The frozen confirmation occupation diagnostic is registered in SUPPLIED-OCCUPATION.md. Run the inputs and generate stages separately to expose exactly the discrete labels available to each pipeline before any scoring. Each stage below uses a single CPU worker.

~~~sh
python -m lattice_symmetry.v3.geometry.supplied inputs
python -m lattice_symmetry.v3.geometry.supplied generate
"$MATCHER_PYTHON" -m lattice_symmetry.v3.geometry.supplied score
python -m lattice_symmetry.v3.geometry.supplied report
python -m lattice_symmetry.v3.geometry.write_coverage_section
python -m lattice_symmetry.v3.geometry.replicated_effects
python -m lattice_symmetry.v3.geometry.check_replication_tables
python -m lattice_symmetry.v3.geometry.official_diagnostics --after-architecture
"$MATCHER_PYTHON" -m lattice_symmetry.v3.geometry.accepted_groups family prior contacts-small-11 contacts-small-23 contacts-small-37 contacts-medium-11 contacts-medium-23 contacts-medium-37
~~~

The official volume analysis requires the exact-prior metadata receipts from `raw_templates` for the analyzed fits. `--available` reads only finished 9,046-target score files. `audits/seed-volume-pattern.csv` contains the number recovered by each volume criterion. The companion `-paired.csv` stores head-to-head gains, losses and target-bootstrap effects versus the identical prior. Stratification preserves the individual candidate when combining a structural match, compatible template and native-volume test.

`python -I lattice_symmetry/v3/geometry/clean_probe.py EXTRACTION MODE OUTPUT.json` checks the model, matcher or symmetry suites in a separately extracted evidence directory (`MODE=model`, `matcher` or `symmetry`). It refuses first-party imports outside that extraction. The model suite instantiates and samples both families, restores the older angular checkpoint and checks its signed Fourier replay. In the full evidence extraction it restores the first frozen contact checkpoint as well. The old Windows matcher installation can require its actual external SciPy directory through `--dependency-site`.
