# CrytalDrift research evidence

The archive uses paths relative to the crystal-drift repository. Each entry in `EVIDENCE-MANIFEST.json` has its uncompressed byte count and SHA-256. Files copied from a separate local research volume appear at their normal repository paths. These include native proposal shards, optimizer states, EMA deployments and ordered CIF catalogs. Frozen training selections and validation panels remain separate files. The reduced uFlowCSP controls use the published Cartesian architecture and retain their own Gaussian initialization law.

## Clean numerical replay

Use a fresh directory to extract the research evidence. The numeric replay uses Python 3.10 or later, NumPy and SciPy. From the extracted repository root run

```sh
python lattice_symmetry/v3/release/replay.py --verify-manifest
```

The replay checks all file hashes. It reconstructs target-wise match rate, scale-enabled matched-target mean best normalized displacement, maximum-site acceptance, native-volume acceptance and seed-paired or family-clustered recovery uncertainty from the ordered outcomes. It verifies matched space-group counts, completed proposals at elapsed deadlines, batch timing records, de novo validity and novelty, fingerprint coverage, density and element-count distances, and the relaxation screen against their contributing records. It writes the reconstructed quantities to `lattice_symmetry/v3/release/numeric-replay.json`. Passing `--panels family shared` restricts CSP replay to the named panels. The protected pilot carries reduced data for a clean packaging test and labels its scope in the manifest.

## Prior releases and retraining

The immutable original evidence ZIP has SHA-256 `644c0456a4f015edb1df00ab766246a1c4922f692bb9329368cd7451c5ec84ec`. The earlier periodic overlay has SHA-256 `985b515251096748b3174e9def43e5d17ccd6ab91e0dc561a2f2e9110eebb0e7`. To reproduce earlier data-preparation pipelines, materialize them in order in a fresh directory and then apply this overlay. The numeric replay runs directly from the full new research evidence ZIP.

```sh
python -m zipfile -e crystal-drift-research.zip work
python -m zipfile -e CrytalDrift-periodic-evidence.zip work
python -m zipfile -e CrytalDrift-v3-research-evidence.zip work/crystal-drift
cd work/crystal-drift
python lattice_symmetry/v3/release/replay.py --verify-manifest
```

The inference and training commands are recorded in `lattice_symmetry/v3/README.md` and `lattice_symmetry/v3/uflow/README.md`. Set `UFLOW_WEIGHTS` to the packaged `lattice_symmetry/v3/uflow/released/ema.safetensors` when using the full published model. That EMA was released by the uFlowCSP authors. The denoiser and transport sources verify against Git commit `e87af8483e6d45b0217b4ede430862e992d2b1d0`. The study matcher uses pymatgen 2024.4.13. Large staged package builds can be recreated with the following command from the original workspace on WSL.

The training GPU used driver 610.88, CUDA 12.4 and cuDNN 9.1.0. The CPU matcher used its own numerical environment. To create the two principal environments from Python 3.11 on an available CUDA-compatible Linux allocation, run the following commands. The three `*-environment.json` receipts record the actual laboratory versions and library locations. The bundled timm sources provide the original import path during native uFlowCSP replay.

```sh
python3.11 -m venv .venv-train
.venv-train/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
.venv-train/bin/python -m pip install numpy==2.3.5 scipy==1.16.3 numba==0.64.0 timm==1.0.21
python3.11 -m venv .venv-match
.venv-match/bin/python -m pip install numpy==1.26.4 scipy==1.10.1 numba==0.59.1 llvmlite==0.42.0 pymatgen==2024.4.13 spglib==2.5.0
export TORCH_PY="$PWD/.venv-train/bin/python" MATCH_PY="$PWD/.venv-match/bin/python"
export MATCHER_PYTHON="$MATCH_PY" UFLOW_WEIGHTS="$PWD/lattice_symmetry/v3/uflow/released/ema.safetensors"
```

```sh
TIMM_SITE=$("$TORCH_PY" -c 'import pathlib, timm; print(pathlib.Path(timm.__file__).resolve().parents[1])')
python lattice_symmetry/v3/release/build_evidence.py /path/to/CrytalDrift-v3-research-evidence.zip --released-ema "$UFLOW_WEIGHTS" --timm-site "$TIMM_SITE"
```

The builder checks scratch payload identities, checks ZIP CRCs and records the finished archive's SHA-256 beside the ZIP. The full profile carries the 207 first-traversal shards and the 1,449 registered later-traversal banks under `lattice_symmetry/v3/models/cache`. Their published digests are in `first-pass-sha256.json` and `fresh-bank-sha256.json`. The preparation code can regenerate the banks from primitive chart data, the training order and the run seeds.

For the first shard of the second traversal, the Linux CPU with the NumPy neighbor fallback reproduced the sampled anchors, source IDs and group assignments. The largest change in a descriptor entry was 7.63 × 10⁻⁶, and the regenerated archive had a different SHA-256. `cpu-bank-replay.json` gives the results for each tensor. The release supplies the original bank bytes for exact comparisons. Re-run this diagnostic with `"$TORCH_PY" -m lattice_symmetry.v3.release.bank_replay bank-replay.json`.

The Linux fresh-bank preparation used Numba 0.64.0 for directed image-neighbor enumeration. With one OpenMP, MKL, OpenBLAS and Numba thread, independent regenerations of `epoch-01/train-194.npz` and `epoch-07/train-206.npz` reproduced the archived SHA-256 values exactly. `numba-bank-replay.json` records the runtime and digests. Use `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 "$TORCH_PY" -m lattice_symmetry.v3.release.bank_replay exact-later.json --exact-later` with the pinned training packages.

The final release includes the standalone arXiv source, its PDF, the two-pass build receipt, author review tables and the earlier reporting milestone. The extracted replay compares every source member of the arXiv package with its counterpart in the research archive. The archive's `.zip.sha256` sidecar identifies the exact bytes, and its receipt records the numeric replay digest after a successful clean extraction. A second workspace copy is made when there is enough room. Otherwise, a workspace pointer records the verified WSL location and digest.

The full release also restores each distinct validation-selected orbit checkpoint, all reduced native selections and the original full-size EMA on CPU. Each restored generator receives two real validation compositions. Run the same isolated check from the extracted project with `"$TORCH_PY" -I lattice_symmetry/v3/release/weights.py "$PWD" selected-weights-replay.json`. The second checksum in the final receipt binds that report.

To stage immutable scientific payloads while the manuscript is being edited, run the builder once with `--stage-dir /path/to/scratch/staging --prepare-stage`. The stage contains the completed weights, fixed raw arrays, ordered CIF archives and datasets. Rerun the builder with the same stage directory, omit `--prepare-stage`, and pass `--verify-fresh /path/to/new/extraction --torch-python "$TORCH_PY"` to append the frozen manuscript, arXiv ZIP and publication manifest. Recovery of an interrupted append checks the compressed prefix digest before restoring the saved central directory.

During this run Ubuntu restarted and cleared its temporary experiment directory. The checksum-indexed scientific stage was in persistent storage. `recover_stage.py` restored 2,648 affected checkpoint and raw-array paths from it and checked their original SHA-256 digests. The earlier source pilot supplied the pinned timm sources. `stage-recovery.json` records the two source digests. A full extracted archive already contains these materials at their canonical paths.
