# What the one-step crystal generator learns

The revised [integrated supplement](paper/supplement.tex) includes the multielement collision and its separating channel, finite-bank proof assumptions and controlled feature ablation. The [numerical witness and gradient checks](revision/ablation/checks.json) exercise the actual descriptor implementations.


The full derivations, proofs, worked crystallographic examples and multielement extension are in Sections S1–S9 of the [integrated manuscript](paper/main.pdf). The [LaTeX source](paper/supplement.tex) includes the assumptions, the finite-bank gradient and the eight-site counterexample. The shorter note below introduces the original binary model.

## Target law and coordinates

A condition $c=(Z_A,Z_B,n_A,n_B)$ specifies the two elements in increasing atomic-number order and their exact site counts in the deposited cell. Given $N$ training records, $N_c$ of which have condition $c$, the target assigns mass $N_c/N$ to $c$ and conditional mass

$$
\widehat P_c=\frac{1}{N_c}\sum_{i:c_i=c}\delta_{x_i}
$$

to the observed cell geometries. Every record counts once. These weights describe database sampling. Temperature, free energy, polymorph multiplicity and synthesis frequency do not enter them. For de novo proposals the sampler separately mixes the empirical joint composition law with independent element substitutions conditional on site counts, in equal proportions. Conditions absent from training have no empirical conditional target. [Data provenance](data/README.md) · [Composition sampler](de_novo.py)

For a row-basis lattice $B$ in ångströms, let $G=BB^\top$, $S=\frac12\log G$, and $k=(S_{11},S_{22},S_{33},\sqrt2 S_{12},\sqrt2 S_{13},\sqrt2 S_{23})$. The network predicts $\tilde k$, standardized using the training mean and population standard deviation in each coordinate. Decoding reverses standardization and gives $G=\exp(2S)$. Each occupied site has fractional row coordinates $f_j\in\mathbb T^3$. A coordinate reference measure for this chart is Lebesgue measure in $k$ times normalized Haar measure on each site torus. The empirical target is atomic, and training evaluates no density relative to this measure.

The generator pushes independent six-dimensional standard Gaussian cell noise and uniform site-torus noise through one network call. Shared site layers receive species labels, cell noise, counts and phases relative to species means. Pooled site features predict the cell. Each site receives a learned bounded shift of its input noise modulo one. Translating all occupied inputs translates all output sites equally and leaves the cell unchanged. Jointly reordering sites and their species labels reorders the output sites. Padding contributes no sites. [Generator and features](model.py) · [Cell coordinates](data.py)

## Geometry presented to the loss

For each of 13 integer wave vectors $h$, define $a_s(h)=n_s^{-1}\sum_{j:z_j=s}\exp(2\pi i\,h\cdot f_j)$. The descriptor $\Phi$ concatenates $\tilde k/\sqrt6$ with the real and imaginary parts of $\overline{a_A(h)}a_B(h)/\sqrt{13}$. For one A and one B, the three unit frequencies recover every component of $f_B-f_A\bmod1$ from its phase. Together with the cell, this identifies a two-site geometry up to common translation **in the deposited basis**. The raw-coordinate AB comparison uses $2((f_B-f_A)\bmod1-1/2)$ in place of those Fourier features.

For binary cells with two to eight sites, the descriptor also includes the two species' squared amplitudes at the 13 waves, scaled by $1/\sqrt{13}$, and six complex triple products, scaled by $1/\sqrt6$. These are $a_A(e_i)a_A(e_j)\overline{a_B(e_i+e_j)}$ and the species-reversed products for $i<j$. Common translation phases cancel. General multisite reconstruction from this finite descriptor is unproved. A change of integral basis $B\mapsto UB$, $f\mapsto fU^{-1}$ preserves the physical crystal but can change both the Gram coordinates and the selected frequencies. The code retains the deposited basis. [Prior work on invariant and incomplete descriptors](prior-art.md)

## The implemented attraction and repulsion

Write $h_i=\Phi(x_i^{\rm gen})$ for a query. Each real positive $p_j$ and independent generated negative $n_j$ have the same reference condition $c_j$. Pooled training draws query and reference records independently with replacement. Within-formula training samples formula blocks with probability $N_c/N$, then uses four fresh queries, four real records sampled with replacement and four generated negatives per block. A singleton formula therefore supplies four copies of its sole positive. [Training loop](train.py)

For each $\tau\in\{0.3,0.8\}$ concatenate the positives and negatives into columns $y_j$. Pooled logits are

$$
L_{ij}=-\|h_i-y_j\|_2/\tau-\tfrac12\|\chi(c_i)-\chi(c_j)\|_2^2,\qquad
\chi(c)=(Z_A/20,Z_B/20,n_A/(n_A+n_B),(n_A+n_B)/4).
$$

Within-formula logits use only the geometry term, separately in each block. Set $A_{ij}=\sqrt{\operatorname{softmax}_{j}(L_{ij})\operatorname{softmax}_{i}(L_{ij})}$. Write $m_i^\pm=\sum_{j\in\pm}A_{ij}$ and $b_i^\pm=\sum_{j\in\pm}A_{ij}y_j$. The field and detached loss are

$$
D_i^\tau=m_i^-b_i^+-m_i^+b_i^-,
\qquad V_i=\tfrac12\sum_\tau\frac{D_i^\tau}{\sqrt{B_q^{-1}\sum_\ell\|D_\ell^\tau\|^2+10^{-8}}},
\qquad \mathcal L=\frac1{2B_q}\sum_i\|h_i-\operatorname{stopgrad}(h_i+V_i)\|^2 .
$$

Here $B_q$ counts **all** queries, including across formula blocks in the RMS. Each $D_i^\tau$ points from the normalized negative centroid to the normalized positive centroid, scaled by $m_i^+m_i^-$. The loss gradient is $-B_q^{-1}\sum_i(\partial_\theta h_i)^\top V_i$. Each query thus contributes its own attraction–repulsion direction through the shared generator Jacobian. The optimizer clips gradients and uses AdamW. The printed loss measures field magnitude on a batch. [Implemented field](model.py) · [Drifting loss precedent](https://arxiv.org/html/2602.04770v2)

## What exchangeability proves

Fix the training archive, parameters, query conditions and query noises, and all reference **condition labels**. Leave the sampled real structures and negative noises random. If the generated and empirical *descriptor distributions* agree at every reference condition, the positive and negative arrays are exchangeable under a simultaneous swap across all blocks. Their columns have matching conditions, so swapping them interchanges the affinities and reverses every $D_i^\tau$. The batch RMS is unchanged. Thus $\mathbb E[V_i\mid\text{fixed information}]=0$, and the expected detached loss gradient is zero because the query Jacobians are fixed. Conditioning additionally on the realized positive records would invalidate that argument in general.

This sufficient condition concerns the gradient of the stated loss. An individual batch can have nonzero drift even at equality. The converse, convergence of training, completeness of the multisite features and thermodynamic stability do not follow.

In the saved float64 check, the maximum descriptor discrepancy after translating and reordering sites was $1.03\times10^{-15}$. Generator equivariance, two-site displacement recovery and pooled reference-swap antisymmetry each had error $1.11\times10^{-16}$. For equal torus displacements of 0.02 across and away from the coordinate seam, the raw feature distances were 1.96 and 0.04. Both periodic feature distances were 0.110055. [Check implementation](verify.py) · [Recorded outputs](geometry-checks.json)
