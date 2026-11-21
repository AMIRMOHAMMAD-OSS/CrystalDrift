# Binary cells from the processed MP-20 release

The split archives are the WyckoffDiff authors’ MP-20 conversion of the DiffCSP++ source, based on Materials Project data (CC BY 4.0). WyckoffDiff trained on WBM. We use its MP-20 data release for this experiment. The recorded versions and compressed SHA-256 checksums are in [manifest.json](manifest.json). Training data are shared with the protostructure audit in [the adjacent data directory](../../crystal/data/). [Data provenance and download links](https://github.com/httk/wyckoffdiff/blob/main/data/README.md)

| Official split | Released rows | Two-site AB | Binary cells with 2–8 sites |
|---|---:|---:|---:|
| Train | 27,136 | 512 | 3,630 |
| Validation | 9,047 | 171 | 1,231 |
| Test | 9,046 | 189 | 1,229 |

Material IDs are unique within each split and disjoint across all splits. The train AB cohort has 472 different element pairs. The corresponding pair occurs in training for 26 validation structures and 32 test structures. Each cohort preserves its structure's original split and material ID; the training and evaluation do not interpret a database count as an equilibrium weight.

## Extracted representation

Each original `cif` column explicitly lists coordinates at full occupancy in space-group P1 with the identity as its sole symmetry operation. All selected rows use site multiplicity one. The published conventional-cell CIF and its inferred space group are excluded from model inputs. Conversion checks the determinant against the deposited volume and round-trips the lattice coordinates. For AB, one site of each species per cell rules out a smaller same-species translation.

The multisite cohort contains the deposited cells. At fractional tolerance $10^{-5}$, a further pure translation maps each species back to itself in five training cells. Two satisfy this at zero reported coordinate error (mp-19006, mp-715262), and three additional cases are mp-1281224, mp-755073 and mp-25234. No validation or test cells satisfy this check. All five remain in the fixed training cohort.

Atoms sort by increasing atomic number with input-row order as the tie-breaker. For AB, the first atom defines the origin, and `f` is the second atom’s wrapped fractional displacement. In the multisite arrays, positions retain their original origin. Missing sites have $Z=0$ and zero coordinates. The model must mask these positions by species. The lattice Gram matrix $G$ is in Å², in the depositor’s basis. Its diagonal gives squared cell lengths. Pack $S=\tfrac12\log G$ as $(S_{11},S_{22},S_{33},\sqrt{2}S_{12},\sqrt{2}S_{13},\sqrt{2}S_{23})$. Matrix exponentiation decodes a positive-definite metric, modulo numerical precision. The representation still depends on basis choice.

To regenerate the arrays, place the training, validation and test archives here, or leave training in the adjacent shared directory. Run `python data.py` from the parent directory. For each split, the AB and multi archives include the original material IDs, source CSV record numbers, species, fractional coordinates, $G$, and packed log stretch.

## Independent environment evaluation

For each structure, compute the volume per atom and shortest A–A, A–B and B–B periodic-image distances, where A has the lower atomic number. Same-species distances include nonzero translations of the same physical site. The vector is invariant to atom order within a species, lattice-vector basis and cell origin. For AB, the two same-species distances coincide, so that coordinate enters the score once. For multisite comparisons, both enter. Held-out inputs supply the atom counts and species fractions.

The search enumerates candidate integer images and certifies the bound with the reciprocal metric. Uncertified cells pass to an exact Cholesky sphere search in three dimensions. An independent check on 45 randomized cells with two, five or eight atoms agreed with direct $[-5,5]^3$ enumeration within $4.5\times10^{-16}$ Å. Integer-basis shear and rotation invariance follow from the distance criterion; the same randomized shear check produced a largest descriptor difference of $1.4\times10^{-12}$. Cells whose predictions exceed the documented numerical envelope or precision condition count as geometric failures. So do distances below 0.5 Å. The validity statistic addresses pair collisions, not thermodynamic stability.

Scores use log descriptors standardized by training-set mean and standard deviation. The per-condition energy score compares the ensemble for a given test formula and atom count against that held-out crystal. It uses the empirical forecast distribution, including diagonal zero distances. The pooled squared RBF MMD uses the biased estimator across conditions separately at each sampled draw index, then averages. The RBF bandwidth is the median Euclidean distance among up to 512 training rows selected with fixed seed 7301. Both discrepancies include physically close contacts if their metrics are numerically evaluable. The report lists numerical failures explicitly. The 15% environment hit requires every physical feature to lie within a factor of 1.15 of its reference. Full crystal equivalence requires a structural matcher.

The finite ensemble budget is 20 AB draws or the first five multisite draws. Consequently, absolute energy scores are best compared between methods in the *same* cohort. Retrieval samples training cells uniformly from exact species-and-count matches when available. Otherwise it keeps species multiplicities when possible and selects training pairs at minimum summed absolute atomic-number distance; tied donors are sampled uniformly. This deliberately simple baseline has access only to training geometry and the supplied held-out composition.
