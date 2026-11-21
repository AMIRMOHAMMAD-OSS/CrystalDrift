# One-call generation from training prototypes

This diagnostic tests how much formula-conditioned recovery comes from the seed and how much comes from the learned angular update. All new angular proposals use the released seed-11 weights at update 256. All train-derived inputs come from the official MP-20 training split. Inference supplies the exact elemental counts in the target cell. The two validation cohorts exclude the original release's 128 checkpoint-selection IDs, and the test subset is fixed in TEST_PROTOCOL.md.

## Proposal routes

| Route | Input structures and update |
|---|---|
| prior | The released fitted cell prior and independent uniform fractional sites. Zero neural calls. |
| angular | The released seed law followed by one angular-generator call. |
| hybrid_prior | Alternate fitted-prior draws with formula-matched training-structure donors. Zero neural calls. |
| hybrid_angular | The same mixed seeds, each processed once by the angular generator. |
| volume_anchor | The same one-call prediction. On donor slots only, rescale the cell isotropically to the input donor's composition-adjusted volume. All new fractional coordinates and cell shape are preserved. |

Donors are ranked first within a matching list of elemental multiplicities in atomic-number order and otherwise among cells with the same site count. Within that candidate set the sorted atomic-number distance sets the order, with seeded random tie breaking. For each donor slot a distinct training material is retrieved. The donor cell obtains an isotropic volume adjustment from the released train-only composition regression. The generator uses the target composition, including when a donor supplies coordinates from another chemistry. When fewer than ten distinct donors exist, the remaining positions in the 20-draw sequence use regular prior samples. The archive records the training index of every donor and a negative sentinel for each regular draw.

The ordinary claim is a match under the release's volume-scaled pymatgen 2024.4.13 RMS matcher. The native-volume endpoint adds a volume ratio between 1/1.1 and 1.1 to a successful match. No target volume or coordinates enter the generator. Each control uses the same random inputs in matched positions. The verification harness reuses predictions across arms. In the proposed sampler, each candidate receives one generator call, then a deterministic cell rescaling on donor slots.

## Replay and records

Run pilot.py generate in the release's CPU PyTorch environment and pilot.py score in its pinned matcher environment. These reconstruct validation IDs 128–255 from the original SHA-256 order. Add --confirmation to both commands for distinct validation IDs 256–383 and to include volume_anchor. Add --test instead for the preregistered 384 official test IDs. Run paired_report.py after scoring the test proposals to produce counts.csv and paired.csv, including the nine-or-more-atom subset and the frozen uFlowCSP replay aligned by material ID. The original per-draw RMS values, native volume ratios and candidate sources are in the NPZ files. Paired uncertainty intervals resample material IDs and condition on the given seed and donor inventory.

Validation replication at 20 draws gave 40/128 volume-scaled and 25/128 native-volume hits with ordinary angular, 99/94 using only the mixed seeds, 101/66 with the frozen angular network applied to the mixed seeds, and 101/93 after anchoring donor volume. These counts are formula-conditioned outcomes with training prototypes. They do not estimate the probability of a novel, stable de novo crystal.

## Prespecified held-out comparison

On 384 MP-20 test targets and 20 proposals, the archived angular checkpoint recovered 120 targets under the volume-scaled matcher and 82 under the native-volume gate. The volume-anchored one-call generator recovered 317 and 296. Released uFlowCSP recovered 304 and 297 on the identical IDs. Paired 95% target bootstrap intervals for anchored minus uFlowCSP were −0.78 to +7.55 percentage points for the ordinary metric and −4.95 to +4.43 for the native metric.

The mixed source alone recovered 317 and 294. At 20 draws three targets gained and three lost a scaled match when the volume-anchored neural update was applied. Three gained and one lost a native match. Among 231 targets with at least nine atoms, the archived angular model recovered 8 scaled matches, compared with 178 from the volume-anchored generator, 178 from the mixed source alone, and 160 from uFlowCSP. The experiment therefore attributes most of the gain to retrieval and transfer of existing structural motifs. It used one frozen angular checkpoint.

Run clock.py with the CPU PyTorch environment for the formula-to-decoded-host-arrays timing. On the first 48 tested IDs with two interleaved repeats and one thread, the ordinary and volume-anchored generators took median 2.05 and 2.41 ms for one complete structure, and 32.08 and 32.88 ms for 20 candidates processed with single-candidate network calls. The zero-call mixed source took 0.63 and 0.79 ms for those workloads. Each episode included a fresh ranking of training donors per target and ended with decoded lattice, fractional and Cartesian arrays. Initial index construction (0.35 seconds), model loading, CIF writing, scoring and relaxation were outside the clock. CPU contention was uncontrolled. These measurements do not cover uFlowCSP runtime.
