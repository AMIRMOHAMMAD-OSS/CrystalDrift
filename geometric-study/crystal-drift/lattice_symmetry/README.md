# A lattice-symmetry-aware drifting model for crystals

The earlier CrystalDrift experiments use a log-metric and Fourier features indexed in the deposited cell basis. They already handle the choice of origin and the ordering of like atoms. A change from basis vectors \((a,b,c)\) to \((a,a+b,c)\) leaves the infinite crystal unchanged but generally changes those features. This prototype changes both the training field and the one-call generator. All predictions in the existing manuscript were produced with the earlier model.

## The training field

Let \(L\) have the three lattice vectors as rows, \(f_i\) be fractional coordinates, and \(z_i\) be atomic numbers. For a physical cutoff \(r_c\), enumerate the periodic neighbors

\[
d_{ijn}=(f_j-f_i+n)L,\qquad n\in\mathbb Z^3,\qquad \|d_{ijn}\|<r_c .
\]

Exclude only the zero image of a site paired with itself. This enumeration includes neighbors across cell faces and multiple images of the same site when appropriate. The implementation bounds each integer index using the norm of the corresponding column of \(L^{-1}\). An oblique cell can require indices outside the nearest 27 image cells.

The descriptor \(\Phi(C)\) consists of log cell volume, element counts, a radial histogram of nonzero lattice translations, and smooth radial histograms of pairs of distinct sites grouped by element pair. Both sides of each comparison use the same ordered vocabulary and cutoff. In each histogram a shell contributes continuously as a distance approaches the cutoff. Atoms of different species remain distinguishable.

The similarity between a query crystal \(C\) and reference \(D\) is

\[
k_\tau(C,D)=\exp\!\left[-\|\Phi(C)-\Phi(D)\|_2/\tau\right].
\]

We use the same normalized attractive and repulsive field as the earlier CrystalDrift implementation, with temperatures \(0.3\) and \(0.8\). The positive bank contains training structures of the specified composition and the negative bank contains independently generated structures. The generator is trained through its descriptor by regression toward a detached feature target. At inference it takes just a seed crystal and its elements.

## The one-call generator

The network passes scalar messages on the periodic neighbor graph. Their inputs include atomic numbers and physical edge lengths. Learned scalar coefficients weight Cartesian edge vectors to give site displacements \(v_i\), and edge dyads \(d^\top d\) together with the identity to give a symmetric strain \(E\). Remove the mean site displacement, then update the atoms and lattice in one forward call:

\[
F_{\rm out}=(F+vL^{-1})\bmod 1,\qquad L_{\rm out}=L\exp(E).
\]

The symmetric matrix exponential gives a nonsingular cell for every finite strain. A fractional-coordinate derivative of a scalar loss is a covector. The generator constructs its site moves as physical Cartesian vectors before converting them to fractions.

## Proposition and proof

Let \(U\in GL(3,\mathbb Z)\) change lattice basis, let \(Q\in O(3)\) change Cartesian frame, let \(P\) permute the sites together with their chemical labels, and let \(t\) shift the fractional origin. Write

\[
L'=ULQ,\qquad F'=(PFU^{-1}+\mathbf 1t^\top)\bmod 1,\qquad Z'=PZ .
\]

**Proposition.** Each query and reference may be independently re-expressed in this way. Every kernel value and the resulting drift of every query in feature space stay unchanged. Applied to equivalent seeds with any fixed weights, the generator produces equivalent output crystals under the same transformation.

**Proof.** An integer unimodular matrix permutes the integer lattice bijectively. For each periodic edge in the original cell there is therefore an edge in the new description with vector \(dQ\), including the integer translations required to wrap its endpoints. The common shift cancels from endpoint differences. The paired edge has the same length and endpoint elements. Cell volume is unchanged because \(|\det U|=|\det Q|=1\). Hence every descriptor and every affinity, displacement in feature space, and normalization computed from the banks has its original value.

The same edge correspondence preserves every learned scalar coefficient of the generator. Thus \(v'=PvQ\) and \(E'=Q^\top E Q\). Substituting in the two update equations gives \(L'_{\rm out}=UL_{\rm out}Q\) and \(F'_{\rm out}=(PF_{\rm out}U^{-1}+\mathbf1t^\top)\bmod 1\). This also proves that any species-preserving space-group operation already shared by the seed's cell and sites remains a symmetry of the output in fractional coordinates. The output may acquire additional symmetries.

## What can be generated under a specified space group

The optional conditional sampler accepts a complete list of operations in a specified setting, a cell volume, and one element per **general-position orbit**. It draws a compatible positive-definite metric by group averaging and expands each random representative into its full orbit. The equivariant generator can then move sites and deform the cell while preserving those operations. For a \(P2_1\) screw axis along \(b\), every site at \((x,y,z)\) has a same-element partner at \((-x,y+\tfrac12,-z)\) modulo integer translations, before and after generation.

This conditional experiment supplies the group and orbit contents. Learning their distribution and occupying special Wyckoff positions are separate tasks. The unconditional smoke test instead draws one physical seed in a chosen basis. The equivariance statement compares two representations of the *same* seed. It also applies in distribution when the prior is pushed forward along the change of representation.

## Evidence and limitations

The tests compare features, affinities, the banked training drift, joint cell and site outputs, and gradients of shared parameters. They include a skew basis, a reflection, independent integer wraps, changes of origin, atom reordering, and a twofold screw operation. An equal-volume cubic and orthorhombic pair provides a control where the descriptor should change. Four optimizer updates on 12 distinct MP-20 training records exercise the training and sampling code. They provide no estimate of generation quality.

The radial descriptor can give distinct structures the same features and it identifies mirror images. The proof covers equivalent choices of basis for one translation lattice. Larger supercell presentations change the recorded volume and counts. Cutoff enumeration can also be expensive in skew cells. Measure wall-clock sampling including neighbor construction before reporting speed.

The next benchmark should compare the trained prototype with the earlier CrystalDrift and a multistep model on identical MP-20 splits and conditioning. For composition-conditioned structure prediction report StructureMatcher match rate, matched RMSD, and timed samples per second. For a space-group-conditioned experiment also report the fraction that retains the requested group and the fraction assigned exactly that group, stating the symmetry-finding tolerance. For unconditional generation also evaluate validity, uniqueness, novelty, and stability.

## Reproduction

From the crystal-drift project directory, with PyTorch, NumPy and SciPy installed:

~~~text
python -m unittest lattice_symmetry.test_symmetry lattice_symmetry.test_spacegroup_seed
python -m lattice_symmetry.training --data scaled/mp20-data/train.npz --updates 4
~~~

The code bundle includes the two reused project modules. The MP-20 training archive is read from the path given on the command line.

## Sources

SymDrift develops both aligned coordinate-space drift and drift in an invariant embedding. The latter groups molecular distances by element pair and motivates the periodic descriptor used here. [Darouich et al., §4.3](https://arxiv.org/html/2605.06140v1#S4.SS3)

Periodic equivariant joint updates of lattice vectors and atomic positions have precedents in crystal generation. [Klipfel et al.](https://arxiv.org/abs/2306.04510)

DiffCSP++ incorporates explicit lattice and Wyckoff-position constraints for generation under a specified space group. [Jiao et al.](https://arxiv.org/abs/2402.03992)
