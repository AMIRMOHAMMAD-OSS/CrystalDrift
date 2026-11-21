# Reproduce the formula-only comparison

Run from the project directory containing the 38-page baseline and `lattice_symmetry`. Original checkpoints, raw pools and evaluation code are read from their existing paths. The new files live in `lattice_symmetry/v2/csp`. The frozen training, bank-sampling and validation rules are in [DESIGN.md](DESIGN.md). The CPU fitting environment used Python 3.10, PyTorch 2.7.1+cpu, NumPy 1.26.4 and Numba 0.59.1. Matching additionally used pymatgen 2024.4.13, spglib 2.5.0 and SciPy. CUDA timing used the Linux WSL build of PyTorch 2.6.0+cu124. The fitted CPU weights can be restored in the newer and older local PyTorch builds. New weight files include optimizer state so training can resume at any archived milestone.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python lattice_symmetry/v2/csp/model.py
python lattice_symmetry/v2/csp/cache.py
python lattice_symmetry/v2/csp/stage.py --overfit
python lattice_symmetry/v2/csp/train.py --seeds 11 --stop 256 --workers 1
python lattice_symmetry/v2/csp/stage.py --seeds 11 --updates 16 64 256
python lattice_symmetry/v2/csp/train.py --seeds 11 23 37 --stop 3392 --workers 1
python lattice_symmetry/v2/csp/stage.py --updates 256 1024 3392
python lattice_symmetry/v2/csp/stage.py --arms prior --updates 0
python lattice_symmetry/v2/csp/lock.py
python lattice_symmetry/v2/csp/predict.py --arms prior angular radial power0 fixed-angular flow16
/path/to/matcher-venv/bin/python lattice_symmetry/v2/csp/match.py --workers 4
```

For a clean rerun, use a fresh copy of the source and baseline inputs. A finished fit skips work after verifying its checksum. Prediction streams are keyed by fit seed and four-target block; each target retains its twenty draws in order. Files are flushed every 256 targets. The crosswalk must be preserved when scoring externally. A deviant number of workers does not change the inference stream. The default full array exports are approximately 860 MB for 18 ordered pools, in addition to the 259 MB reusable all-training positive descriptor array. A one-thread CPU process suffices for each fitting or generation task. The reference clock requires the GPU specified below and the 525 MB released uFlowCSP weight file. Restrict concurrent workers on a shared-memory machine.

```sh
python lattice_symmetry/v2/csp/timing.py --methods retrieval old-drift meanflow old-flow16 uflow \
  --weights /path/to/ema.safetensors --site /path/to/timm-site
python lattice_symmetry/v2/csp/timing.py --methods prior radial power0 angular fixed-angular flow16 \
  --site /path/to/numba-and-llvmlite-site
python lattice_symmetry/v2/csp/interleave.py --site /path/to/numba-and-llvmlite-site
/path/to/matcher-venv/bin/python lattice_symmetry/v2/csp/interleave_score.py
/path/to/matcher-venv/bin/python lattice_symmetry/v2/csp/gpu_score.py
python lattice_symmetry/v2/csp/sources.py
python lattice_symmetry/v2/csp/lock.py
python lattice_symmetry/v2/csp/verify.py
python lattice_symmetry/v2/csp/cpu_clock.py
python lattice_symmetry/v2/csp/training_figure.py
python lattice_symmetry/v2/csp/report.py
python lattice_symmetry/v2/csp/paper.py
```

The geometry interface for a separate de novo composition sampler is `lattice_symmetry.v2.csp.model.load_service(arm, seed=11, device="cpu")`. The returned function accepts a batch of zero-padded, ascending species rows and a NumPy Generator. It returns packed six-component half-log Gram coordinates and wrapped fractional coordinates. The prior-only arm returns the sampled seed directly. Call `checkpoint_path(arm,seed)` to identify the exact selected artifact, including the prior file itself for the zero-call control. GPU timings include encoding and decoding, with the per-episode host arrays and cumulative times in `gpu/<arm>-episodes.npz`. The dedicated 10,000-structure workloads include warm output materialization and retain each batch duration and count. Comparing the wall clocks of archived and current releases requires accounting for competing host work and the preserved run settings.
