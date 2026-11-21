# Can composition pooling distinguish two decorated crystals?

Two MP-20 training structures have the same Mg₄Zn₈ conventional cell composition and space group 63 (Cmcm). A sum or mean of element embeddings gives them the same value, even if the encoder also receives the element counts and space group. Counting symmetry orbits per element distinguishes them. A change of origin can relabel equivalent positions but preserves the number of distinct orbits of each species.

| Training CSV record | Material | Mg assignment | Zn assignment | Orbit counts (Mg, Zn) |
|---:|---|---|---|---|
| 237 | mp-1094439 | one 4c | two 4c | (1, 2) |
| 2,514 | mp-1094198 | one 4c | one 8g | (1, 1) |

The records are one-indexed after the CSV header. Quoted CIFs span multiple text lines. Each 4c orbit contributes four atoms in the conventional cell. The 8g orbit contributes eight. Both assignments obey the same composition. [WyckoffDiff processed MP-20 release](https://github.com/httk/wyckoffdiff/blob/main/data/README.md)

## Complete training-file audit

We grouped records by their exact conventional-cell element counts and space group. For each species, we summed the number of occupied orbits encoded in the release’s `wyckoff_spglib` descriptor. A coefficient on a letter counts separate orbits of that type. The parser checked element names against the conventional CIF formula, conventional atom count against the Pearson symbol, and space group against the supplied conventional space-group column.

| Slice | Rows | Distinct group keys | Keys with multiple rows | Keys with differing orbit profiles | Unordered row pairs with differing profiles |
|---|---:|---:|---:|---:|---:|
| First 2,000 records | 2,000 | 1,995 | 3 | 0 | 0 |
| Entire training split | 27,136 | 26,635 | 348 | 57 | 100 |

These counts describe the processed training release. Other genuinely distinct decorations can share a per-species orbit count vector. The tally also depends on the symmetry assignment in the spglib preprocessing. It does not measure the prevalence of polymorphism or the performance of a generator.

## Gradient check

Hold the Mg orbit fixed and allow Zn to use the observed 4c and 8g orbit types. Let the Zn repeat counts be $c$ and $g$. Conditioning on $4c+8g=8$ leaves $(0,1)$ and $(2,0)$. Give each allowed pair a score equal to the sum of its two categorical count logits, then normalize the two scores with a softmax.

The expected orbit-count feature is $(1,\mathbb{E}[c+g])$. At a one-hot state distribution, it agrees exactly with the dataset’s hard-count feature for the corresponding crystal. Equal logits give the midpoint $(1,1.5)$. With half the squared Euclidean distance to $(1,2)$ as the loss, the gradient is $(0.125,0,-0.125,-0.125,0.125)$ in logit order $c=0,1,2$ and $g=0,1$. The unreachable $c=1$ state receives zero gradient. Central differences with step $10^{-5}$ agree to $8.2\times10^{-13}$. Expected atom counts remain $(4,8)$ and their derivatives vanish.

This checks a differentiable feature on two observed assignments. Enumerating all globally admissible configurations for an arbitrary composition, learning a useful feature space, refining a pretrained denoiser, decoding continuous geometry and validating relaxed structures remain open.

## Reproduction

Run from this folder with Python 3.10 or newer and PyTorch. The recorded run used CPU PyTorch 2.7.1 with float64. The [included training archive](data/mp20_train.csv.bz2) is the unmodified MP-20 release processed by the WyckoffDiff authors from Materials Project structures. The release credits Materials Project and specifies [CC BY 4.0](https://github.com/httk/wyckoffdiff/blob/main/data/README.md#mp20). The program checks the bzip2 end marker and writes both counts and gradient checks to `orbit_probe.json`. The compressed file has 8,785,424 bytes and SHA-256 `aa9ccf4cf83a58e2aba3492e12f0eaf75bdf2946a7c90fa5bbeea94d731b721a`. [Training archive](https://public.openmaterialsdb.se/wyckoffdiff/data/mp20/raw/mp20_train.csv.bz2)

```shell
python orbit_probe.py data/mp20_train.csv.bz2
```

The [single-update loss check](reading-guide.md#what-we-checked-on-real-structures) uses these saved features and reports separate results for both embeddings.
