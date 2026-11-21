# Occupied Wyckoff template experiments

Run from the CrytalDrift repository root. The original 38-page release is a separate, fixed-basis baseline. This directory evaluates protocol C using the actual v2 physical generator. The network used on expanded coordinates is the same class used in the formula-only angular experiment. The template sampler sets the sites' orbit constraints and draws the compatible cell without a supplied reference volume.

## Inputs and dependencies

`scaled/mp20-data/{train,val,test}.npz` supply the official geometry splits. Symmetry labels use spglib 2.5.0, with a native-to-conventional primitive-site crosswalk and the released `pymatgen==2024.4.13` matcher. The main local fits used PyTorch 2.7.1, NumPy 1.26.4 and SciPy; code uses one PyTorch CPU thread. CHGNet 0.3.8 with its 0.3.0 weights, ASE FIRE and the parent physical-screening environment generate the relaxed samples. The external CUDA replay used PyTorch 2.6.0+cu124 on the RTX 3060 Laptop GPU and FP32 tensors with TF32 multiplication enabled. Matplotlib and ASE provide plotting and CIF exchange. The operator/label APIs are defined in the [spglib dataset documentation](https://spglib.readthedocs.io/en/v2.5.0/dataset.html). The standard-setting chart source and its license are preserved with the generated chart archive. [PyXtal 1.0.9 release](https://pypi.org/project/pyxtal/1.0.9/)

## Label extraction, fitting and output

`charts.npz` contains 1,731 positions in 230 Hall settings. To rebuild it from the pinned wheel, download the wheel and run `extract_charts.py WHEEL`. The wheel hash is in `chart-source.json`. Label shards are split into 200 consecutive official rows and can resume after interruption. The labels store the species/letter template, fitted continuous coordinates for the training/reference target, conventional cell, primitive-site maps, origin shift, lattice transformation and reconstruction discrepancy. Only the group number in the fixed Hall setting and the unordered species/letter pairs are available to generation. `prior.py` accepts these two arguments. The 128-target panels retain all selected IDs, while the supplementary single-draw test retains all 9,046 official IDs including failed template round trips. The reduced check at stricter and looser label-detection tolerances uses 128 validation and 128 test records. `label-systems-dof.csv` counts occupied orbits, rather than counting a mixed-degree target just once.

~~~sh
W=lattice_symmetry/v2/wyckoff
python "$W/labels.py" --workers 3
python "$W/cohort.py"
python lattice_symmetry/v2/csp/cache.py
python "$W/train_template.py"
python "$W/train_template.py" --orbit
python "$W/test_templates.py"
python "$W/evaluate.py" prior
python "$W/evaluate.py" angular-C
python "$W/evaluate.py" orbit-C
python "$W/evaluate.py" angular-A
python "$W/full_test.py"
~~~

`angular-A` uses the validation-selected seed-11 formula-only angular generator. Run the A training and selection guide first when rebuilding this checkpoint. The v2 C objective shares the same train-derived feature block scales. `train_template.py --out PATH --stop 64`, followed by `train_template.py --out PATH --resume --stop 256`, exercises a deterministic interrupt/resume. The separate `restart.json` checks the resulting weights against the selected published run. Run `verify.py --restart PATH/angular-template-256.pt` to regenerate that test. The raw pools have material-ID and ordered draw axes, conventional cells, fractional coordinates and a mask encoded by zero atomic numbers. Native group labels remain in the target crosswalk. Each run receipt identifies its exact weights and pool checksum. `template-tests.json` includes analytic regression cases before and after an optimizer update and reindexing of repeated same-species orbits. `scores/*.jsonl.gz` preserves every draw, failed match, matcher RMS, structural validity, tolerance-specific subgroup, exact group, and independent metric/orbit residual.

## Independent DiffCSP++ comparator

The published weights have SHA-256 `8d7615c73e4f55e52c310122a0e194057d0ccff252255a6497cbed41ca624142`. The released configuration belongs to epoch 1000, training step 212000. The numeric tensor bundle and hash-checked sampling/decoder sources are sufficient for both reported replays. Only group, template letters and atom types are supplied to the native sampler. `diffpp.py` strictly loads decoder weights, lattice-family buffers and learned scheduler buffers. It executes the authors' sampling method with a native PyTorch scatter mean. A saved CPU parity check compares complete emitted structures when the fully connected graph and repeated sine embedding are cached or recomputed for every layer. The 20-iteration replay uses original-index time embeddings and resampled marginal noise buffers, and the native 999-iteration control passes the original noise buffers unchanged. Each iteration calls the decoder twice. The release makes no claim to reproduce the published Orbitol SPARC or SUN benchmark. [Released DiffCSP++ repository and API](https://github.com/jiaor17/DiffCSP-PP/tree/e82e7ffa7cb2a383bde69b067a343ca137d73e47) · [Published sampling paper](https://arxiv.org/abs/2402.03992)

~~~sh
W=lattice_symmetry/v2/wyckoff
python "$W/diffpp.py" --device cpu --parity --steps 5 --split val
python "$W/diffpp.py" --device cuda --steps 20 --split test --count 128 --k 20 --batch 16 --tf32
python "$W/diffpp.py" --device cuda --steps 999 --split test --count 128 --k 1 --batch 16 --tf32
python "$W/evaluate.py" DiffCSPpp-S20 --score-only
python "$W/evaluate.py" DiffCSPpp-S999-test-128x1 --score-only
~~~

The external replay saves progressive checkpoints after four batches with original material and draw order. The output is sealed only after all supplied templates finish. To rebuild the tensor bundle, checkout the commit above, run `fetch_diffpp.py PATH/diffpp.ckpt` and `prepare_diffpp.py PATH/diffpp.ckpt CHECKOUT`. The latter checks the checkpoint, Git commit and tensor strides, and uses an inert-metadata reader. The four directories under `external` contain the snapshot/license, source receipt and learned weights.

## Evaluation and manuscript

~~~sh
W=lattice_symmetry/v2/wyckoff
python "$W/relax.py" prior angular-A angular-C orbit-C DiffCSPpp-S20 DiffCSPpp-S999-test-128x1 --workers 2
python "$W/timing.py"
python "$W/operation_order.py"
python "$W/report.py"
python "$W/example.py"
python "$W/paper.py"
python "$W/verify.py"
~~~

The two rendered figures also have editable SVGs and selection/plot data. `retention.csv` gives the raw/post results by tolerance with target-clustered uncertainty. In `strata.csv`, a target with different site dimensions contributes to each dimension it contains. `relaxation` includes unfiltered first-draw inputs, status, forces, number of steps, wall time, and raw/final CIFs. The physical observable is group retention under an unconstrained relaxation with no stability ranking or hull claim. The matched CPU clock includes representative sampling, metric projection, expansion, the declared neural calls and decoded arrays on the host. It starts after warm-up with models and charts resident, excludes target labelling, matching and relaxation, and shares the host with other CPU tasks. Separate external GPU runtimes refer to the actually archived raw banks. Orbit templates and their corresponding continuous target structures are never mixed into the released formula-only benchmark.
