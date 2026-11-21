# Resource-matched Cartesian uFlowCSP

We retrain the authors' Cartesian transformer from scratch on the 26,614 MP-20 training IDs registered in the shared protocol. Every forward uses 100 padded tokens, the native per-token Cartesian position and nine repeated cell entries, learned species identities, four fixed elemental attributes, a global formula embedding, and the two MeanFlow times. The authors' model, transport and timestep/attention implementations are staged at runtime after checking their Git blob hashes. The reference checkpoint stays a separate released model. [Pinned Cartesian source](https://github.com/usccolumbia/uFlowCSP/tree/e87af8483e6d45b0217b4ede430862e992d2b1d0) · [Original paper](https://arxiv.org/abs/2609.09799)

| Model | Width | Depth | Heads | Registered parameters | Active parameters |
|---|---:|---:|---:|---:|---:|
| uFlowCSP-small, retrained by us | 64 | 2 | 4 | 233,868 | 219,084 |
| uFlowCSP-medium, retrained by us | 160 | 4 | 8 | 2,135,052 | 2,098,092 |
| Released uFlowCSP | 768 | 12 | 12 | 131,215,116 | 131,037,708 |

The registered architecture always allocates a 231-entry fine-space-group embedding. Its forward path is disabled. The effective counts subtract precisely that table. Seven crystal systems and the null embedding remain active. The optimizer carries a separate training-time exponential moving average of the model. Inference evaluates the EMA once, using space group zero. Constructor defaults are overridden by the released training script, including 80% classifier-free label dropout, the 0.25 flow ratio, guidance scale 1.5, and CFG ratio 0.8. The alternate torus transport and structure-level lattice prior are disabled. All generated proposals undergo the same host decoding, and the Gaussian prior control bypasses the transformer.

The native training state is Niggli-reduced, then reconstructed in the authors' Cartesian representative. Canonical ordering uses the published 0.1 Å symmetry algorithm. The group used to derive the training-only crystal-system token comes directly from spglib at 0.1 Å. The author's preprocessing additionally refines the cell and asks PyXtal to identify the group. All 26,614 coarse classes in this run agree with the independent MP-20 CSV class. CrystalNN edges computed by the published dataset pipeline never enter the denoiser or the loss. This loader skips their construction. The held-out formula path sorts the supplied elements by electronegativity and only ever receives the null crystal-system signal.

The short fits use AdamW with learning rate 0.0003, betas (0.9, 0.95), no weight decay, batch size 32, norm clipping at one, full-precision JVPs and EMA decay 0.99. The release used a much longer training horizon, rate 0.0001, batch size 256, bf16 mixed precision and default 0.9999 EMA. We retained the authors' native detached MeanFlow target and adaptive loss, calling their `MeanFlowTransport.loss` directly. One traversal has 832 updates over 26,614 structures and 277,022 atoms, split into two halves of 13,307. Every fit saves its actual updates, atom count, distinct exact compositions, gradient norms, memory and wall time. The first and last updates of each half use the same target and held-out data rules. The separately registered longer exposure continues the seed-11 optimizer and EMA through eight traversals.

## Replay

Set `UFLOW_TIMM` to a directory containing timm 1.0.21 if it is absent from the CUDA environment. `UFLOW_WEIGHTS` names the earlier Figshare-derived `ema.safetensors` when replaying the released full-size model. Both reduced tiers start from random weights. The commands run from the crystal-drift repository root. The file supplied to the preparation step must be the official CSV with SHA-256 `aa9ccf4cf83a58e2aba3492e12f0eaf75bdf2946a7c90fa5bbeea94d731b721a`. The fixed release evidence archive contains this file.

```sh
U=lattice_symmetry/v3/uflow
python lattice_symmetry/v3/protocol/lock.py
"$MATCHER_PYTHON" "$U/prepare.py" /path/to/mp20_train.csv.bz2 --workers 3
python "$U/train.py" small --batch 32
python "$U/train.py" medium --batch 32
python "$U/generate.py" small dev
python "$U/generate.py" medium dev
"$MATCHER_PYTHON" "$U/evaluate.py" small-11 dev
"$MATCHER_PYTHON" "$U/evaluate.py" medium-11 dev
python "$U/generate.py" small frozen
python "$U/generate.py" medium frozen
python "$U/generate.py" prior frozen --device cpu
"$MATCHER_PYTHON" "$U/score_queue.py" small-11 medium-11 prior-11 --workers 2
python "$U/replicate.py"
python "$U/extend.py"
"$MATCHER_PYTHON" "$U/evaluate.py" small-11-long dev
"$MATCHER_PYTHON" "$U/evaluate.py" medium-11-long dev
python "$U/generate.py" small frozen --extended
python "$U/generate.py" medium frozen --extended
"$MATCHER_PYTHON" "$U/score_queue.py" small-11-long medium-11-long --workers 2
python "$U/audit_inputs.py"
```

For the seed-23/37 confirmations use `evaluate.py small-23 dev` followed by `generate.py small frozen --seed 23 --panels confirm`, then `evaluate.py small-23 frozen --panels confirm`. Replace the tier and seed to repeat. Training automatically restores each atomic optimizer/RNG checkpoint. The generator skips verified completed 256-target shards and checks their formula IDs and checkpoint identity. A fresh output tree gives an independent rerun.

The exact common-ID comparison uses the orbit model's 26,419 training IDs and retained shard ordering. After preparing `models/cache/inventory.json`, the following sequence fits both comparator tiers, locks the registered development looks, regenerates the native one-call proposals and reconstructs the reported intervals. `final_gpu.py` requires the frozen empirical composition list in `denovo/pools/prior.npz` and the published EMA path in `UFLOW_WEIGHTS`.

```sh
python "$U/window.py" fit/small/11/common fit/medium/11/common
"$MATCHER_PYTHON" "$U/lock_available.py"
python "$U/final_gpu.py"
"$MATCHER_PYTHON" "$U/finish_cpu.py"
python "$U/results.py"
python "$U/prior_law.py"
python "$U/native_symmetry.py"
python "$U/curves.py"
```

After the seed-11 exposure series and isolated clocks had closed, the previously frozen one-traversal seed-23/37 checkpoints were applied to both protected family panels and all 9,046 official targets. Training data, hyperparameters, candidate counts, native match settings and validation choices were held fixed. The proposal RNG depends exclusively on the dataset split and original row index, so every training seed receives the same ordered twenty raw Gaussian draws. Each replicate shard stores the digest of all 100 padded tokens and twelve scalar channels before inference. An independent CPU generator reconstructs the noise, checks each digest and reruns the identity decoder against the original zero-call arrays. `native-three-seed.json` averages outcomes across training seeds within each target before bootstrap resampling. Family panels resample the operational family.

```sh
python "$U/frozen_replicates.py"
"$MATCHER_PYTHON" "$U/score_queue.py" small-23 medium-23 small-37 medium-37 --panels family shared test --workers 4
python "$U/stream_identity.py" --verify
python "$U/replicate_summary.py"
python "$U/results.py"
```

`skew_check.py` verifies the severe-skew origin-image enumeration on the full ordered lattice vectors and distances. A unimodular LLL map indexes the same physical points, which are filtered in the original Cartesian arithmetic and restored to the pinned C routine's linked-cell visit order. All 4,272 ordered images in 100 stock comparisons agreed exactly. The archived prior draw 19 for mp-1183663, with native condition number 52,879, then returned both matcher nonmatches in 0.082 seconds. That draw remains among the original twenty. [Pinned pymatgen neighbor search](https://github.com/materialsproject/pymatgen/blob/v2024.4.13/pymatgen/optimization/neighbors.pyx)

The fit receipts live under `fits`. Each `predictions` shard stores original material IDs, native official row numbers, enough atomic numbers to decode all ordered draws, raw lattice rows and fractional sites. The matching output stores RMS, maximum-site decision, native log volume ratio and failure code for every attempt. A shard checksum binds the score to its input. The two independent validation panels and two protected family challenges use the same specification as the drift experiments. The released external checkpoint saw the original training archive, which includes protected family challenge examples. Its published/full-split test recovery and a same-device, formula-only timing replay remain separate from the protected-family comparisons.

## Gaussian controls for the published checkpoint

The published-checkpoint comparison has a second zero-call pool. Its original 128-target, five-draw prefix used CPU PyTorch 2.7.1, a single seed of 2026, two threads and batches of 12 in target-major order. The other 180,280 draws used CPU Gaussian noise created by PyTorch 2.6.0+cu124 and CUDA decoding. Each 24-target minibatch starts its own RNG with `2026000000 + 1000000*draw + first_target_index`. The initial 128-target tile has its own boundaries. Every subsequent tile contains up to 256 targets and restarts minibatching at its left boundary. These historical draws are distinct from the per-row seeded reduced-model experiment. On both branches the control skips the learned forward and follows the published log-Gram and fractional-site decoding path. The new pinned matcher includes all positive native volumes.

Run the prefix in the original CPU environment with `UFLOW_WEIGHTS` pointing to the published safetensors file, then use the CUDA PyTorch environment for the remaining rows. Both stages also compare sparse full-EMA reconstructions against the archived lattice and site arrays. The per-draw SHA-256 values in the zero-call pool bind the ordered Gaussian tensors to each target.

```sh
python3.10 -m venv .venv-uflow-prefix
CPU_271_PY="$PWD/.venv-uflow-prefix/bin/python"
"$CPU_271_PY" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1
"$CPU_271_PY" -m pip install numpy==1.26.4 scipy timm==1.0.21
"$CPU_271_PY" "$U/published_prior.py" prefix
"$CPU_271_PY" "$U/published_stream_check.py" prefix
"$TORCH_PY" "$U/published_prior.py" gpu
"$TORCH_PY" "$U/published_stream_check.py" remainder
"$MATCHER_PYTHON" "$U/score_queue.py" published-prior --panels test --workers 4
python "$U/published_effect.py"
```

## Tiny positive volumes and skew cells

The native identity sampler can return sub-0.1 Å³ cells. They remain in the scale-enabled evaluation. When the primitive target already has as many sites as the native proposal, any match also requires the proposal to remain primitive at that size. Its physical lattice must admit a unimodular frame in the matcher's length and angle ranges. After rescaling both metrics to unit volume, the inherited lattice and fractional site feasibility check enumerates the admissible frames. An impossible frame certifies a failed match for every positive volume. Passing cases and unmatched primitive sizes go to the public API in native units. For a skewed input, the certificate first checks an integer-unimodular LLL change of basis and transports its fractional sites. It defers to pymatgen for original condition numbers above 10 million, reduced condition numbers above 1,000, and randomly selected certified exclusions. Forwarded public calls see the original native cell. `check_tiny.py` compares the accelerated and public answers for 58 fixtures, including isotropic contractions, integer shears, native Gaussian cells, and both positive and negative outcomes. The smallest verified volume was 0.000027 Å³ and maximum tested condition number was 2,428. An exploratory cubic cell of side 0.001 Å exhausted the available working memory in pymatgen's sphere search during its direct check. The pinned sphere implementation normally enumerates a sphere of at least 1 Å and discards points beyond the requested radius. We pass min_r=0 to this compiled routine for positive radii below 1 Å and apply its identical final distance mask. This changes the enumeration bound. All 80 independent image-label and distance checks and all 480 native confirmation outcomes agreed. Using that enumeration bound, the same cubic 1e-9 Å³ stress fixture returned a scale-enabled and maximum-site nonmatch in 0.55 s. Its unmodified memory-exhausted path still has no returned decision. See sphere_check.py, sphere-search-parity.json and relative-radius-parity.npz.


