# MP-20 formula-only gap diagnostics

These post-release analyses compare angular CrytalDrift with the released one-evaluation uFlowCSP. All reported targets retain their official MP-20 IDs. The two generators receive the elements and their counts in the native cell. Neither receives a target group, template or volume in this task.

| Sites in the native cell | Targets | Angular match at 20 draws | uFlowCSP match at 20 draws |
|---|---:|---:|---:|
| 1–4 | 1,852 | 91.50% | 95.36% |
| 5–8 | 2,085 | 48.76% | 91.37% |
| 9–12 | 2,193 | 6.19% | 83.77% |
| 13–20 | 2,916 | 0.97% | 57.17% |

Angular rates average three training seeds. uFlowCSP uses one released checkpoint and one ordered set of candidates. The 5,109 cells with nine or more atoms account for 77.68% of the total net hit deficit. At draw one, overall match rates are 6.83% and 46.32%. At draw 20, they are 31.79% and 79.32%. An additional native-volume gate (a ratio between 1/1.10 and 1.10) gives 21.24% and 78.07%. A distinct strict-maximum-site fit check gives 17.81% and 67.78%. The default matchup allows volume scaling. On the 2,744 targets matched by both seed 11 and uFlowCSP at 20 draws, the mean best normalized displacement is 0.342 versus 0.033. This comparison uses identical targets.

On seed 11, draw one, the generator moved each atom a median 0.42 Å from its coupled random prior site (90th percentile 0.68 Å). The median volume change was +11.4%. These are wrapped-chart displacement lengths in the input physical cell. The starting positions were independent uniform fractional draws.

## Component swaps

For the first 512 targets in the release's hash order, we used seed 11 and the first draw from each model. We paired lattice Gram coordinates from one model with the entire fractional motif of the other. All rows used the standard primitive, scale-enabled, species-preserving pymatgen matcher. The original model scores were independently matched and agreed with the archive.

| Lattice | Fractional motif | Hits / 512 |
|---|---|---:|
| CrytalDrift | CrytalDrift | 38 |
| uFlowCSP | CrytalDrift | 65 |
| CrytalDrift | uFlowCSP | 50 |
| uFlowCSP | uFlowCSP | 222 |
| Reference | CrytalDrift | 107 |
| CrytalDrift | Reference | 62 |

The reference rows are retrospective interventions. Cross-model or reference swaps exchange components in the models' stored coordinate charts and can disrupt correlations between lattice and sites. These interventions measure whether a single substituted component suffices in this convention. They do not attribute a causal percentage of the gap to each component or define deployable oracle results.

## Group detection

The same first 512 targets and first draws were assessed with spglib 2.5.0. At symprec 0.1 Å the angular outputs were assigned P1 in 510 cases and the uFlowCSP outputs in 314. Reference group numbers agreed in 8 and 96 cases, respectively. At 0.01 Å, the P1 counts were 510 and 499, with 11 reference-group agreements in both arms. Reference P1 counts were 8 and 11 at the two tolerances. All detected groups were available. These results concern spontaneous symmetry of raw formula-only outputs and are separate from the supplied-template experiment.

## Reproduction

Use the original research archives and their directory layout, including the external draw files, per-model ordered outputs, test and train data arrays. Run the following from the crystal-drift project root. The first two scripts need NumPy. The other two use the matching environment pinned by the main study, including pymatgen 2024.4.13 and spglib 2.5.0.

~~~sh
python lattice_symmetry/v2/diagnostics_gap/paired_gap.py
python lattice_symmetry/v2/diagnostics_gap/prior_moves.py
python lattice_symmetry/v2/diagnostics_gap/component_swaps.py
python lattice_symmetry/v2/diagnostics_gap/group_diagnostic.py
~~~

The scripts independently locate the project relative to themselves. Output tables in this folder include exact target IDs for the 512-case controls and per-seed paired-hit counts for every full-cohort stratum. The site-count and composition subdivisions are descriptive post-release checks. Training support refers to occurrences of element-count vectors (or their explicitly reduced forms) in the local MP-20 training split. The analysis does not audit a third party's original training pipeline.
