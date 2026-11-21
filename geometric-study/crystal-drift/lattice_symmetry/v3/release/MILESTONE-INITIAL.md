# Overnight research milestone

2026-11-21T05:06+03:30

Each table entry covers all 9,046 official targets and preserves the order of its 20 proposals. Only complete target-level crosswalks enter this snapshot.

| Pipeline | MR@5, count | MR@20, count | Native@5, count | Five-draw gain over symbolic prior, percentage points (95% interval) |
|---|---:|---:|---:|---:|
| angular-small-11 | 5,176 | 6,022 | 3,420 | +4.21 (+3.73, +4.70) |
| contacts-small-11 | 5,176 | 6,022 | 3,624 | +4.21 (+3.73, +4.70) |
| prior | 4,795 | 5,610 | 3,970 | +0.00 (+0.00, +0.00) |
| reciprocal-small-11 | 5,172 | 6,022 | 3,150 | +4.17 (+3.67, +4.65) |
| released-uflow | 6,156 | 7,175 | 6,042 | +15.05 (+13.95, +16.11) |

The primary descriptor screen trained 218,497-parameter angular, inter-orbit contact and reciprocal updates over the same 26,419 anchored primitive structures. They each recovered 63/96 development targets and 50/96 on independent confirmation at their two-look selections. The identical prior recovered 59 and 48. Both 2,098,297-parameter drift and same-backbone MeanFlow have completed a first traversal. The downsized native uFlowCSP models finished eight traversals of 26,614 structures, and separate one-traversal fits used the exact common 26,419 anchor IDs.

With 24 prototype families withheld, the first-traversal small contact map recovered 49/228 at five draws, against 46 for its identical prior. The paired family-bootstrap interval was 0.00 to 3.23 percentage points. None of the 49 best accepted structures had the target's detected group at 0.01 Å. The same update gained ten targets among 96 held-out chemistries in represented families. The 25 exploratory cases without a continuous anonymous analogue yielded one identical hit for every frozen arm at five draws.

The small contact model generated 8,883 structurally valid proposals and 4,359 jointly valid, unique and training-novel proposals from 10,000 empirical formulas. The identity prior yielded 8,107 and 4,140. In the paired eight-case CHGNet screen, the contact model yielded zero structures satisfying every restricted-hull gate. Its zero-success upper limit is 31.2% under independent Bernoulli sampling.

The served seed distribution and physical-volume behavior materially affect interpretation. A single neural evaluation counts work per proposal. The synchronized CPU/GPU clock, remaining replications and extended training budgets are recorded separately when complete. Training and scoring continue after this milestone.
