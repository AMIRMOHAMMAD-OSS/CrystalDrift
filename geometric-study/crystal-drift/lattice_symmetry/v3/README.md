# CrytalDrift build, train and replay

The author-review manuscript is `release/CrytalDrift.pdf`. The minimal upload source is `release/CrytalDrift-arxiv-source.zip`. `release/MORNING.md` gives the paired scientific conclusion. `release/MAIN-COMPARISON.csv` joins the public test comparison to its fixed training exposure, warm first-proposal timings and 10,000-structure workload. `release/SEED-RESULTS.csv` contains the three independent fits at each capacity and their paired initializers. `release/CLAIM-AUDIT.md` points each reported endpoint to its source. The seven ordered CIF archives in the research evidence contain the structures behind the de novo plots.

From an extracted research evidence tree, run the following commands at the repository root. The numerical replay reads the deposited outcomes and reconstructs the panel results. The manuscript rebuild re-runs the statistics and plots and compiles twice from a fresh extraction of the minimal source ZIP. Use the analysis environment with NumPy, SciPy, Matplotlib, Pillow and PyTorch, plus command-line pdfLaTeX and Poppler. Under Windows the same relative commands work with an active Python environment.

~~~sh
python lattice_symmetry/v3/release/replay.py --verify-manifest
python lattice_symmetry/v3/paper/rebuild.py --analysis
python lattice_symmetry/v3/paper/check_claims.py
~~~

This study adds a capacity-controlled orbit refiner to the periodic release in `../v2`. Commands below start in the **crystal-drift repository root**. The earlier fixed-basis and periodic archives supply the MP-20 arrays, chart data, canonical atom-ordering tables, pinned matching implementation and released baselines. The full new study uses only the records enumerated in `lattice_symmetry/v3/protocol/splits.json` for fitting. Keep the subdirectory at this relative path.

## Environments and data

Training was run on a 6 GiB NVIDIA GeForce RTX 3060 Laptop GPU under WSL2 using PyTorch 2.6.0+cu124, with full-precision weights and arithmetic. The workstation also has an Intel Core i7-12700H. Set `TORCH_PY` to the appropriate GPU interpreter. Set `MATCH_PY` to the Python 3.10 environment with NumPy 1.26.4, SciPy 1.10.1, Numba 0.59.1, pymatgen 2024.4.13 and spglib 2.5.0 used to score predictions. The example Linux GPU installations in the previous release use `mp_lstm` for torch and the supplemental Numba and llvmlite packages documented in `lattice_symmetry/v2/csp/local-runtime.json`. The equivalent Linux worker must pass the public matcher parity tests. CUDA-training claims and published timing numbers are machine-specific.

The reduced uFlowCSP imports the authors' released source commit and chemistry inputs, identified under `revision/full/external`. Its native transport requires timm 1.0.21. Set `UFLOW_TIMM` to the directory containing that package if it is outside the environment. For the full-size reference, set `UFLOW_WEIGHTS` to the released EMA Safetensors file specified in `revision/full/external/README.md`. Setting these variables loads existing local files. De novo scoring uses the SMACT 2.5.5 and matminer 0.9.2 installations under `CRYSTAL_METRICS_SITE`. The relaxation worker uses the CHGNet 0.3.0 weights through the 0.3.8 package and ASE FIRE. Previously released resources are independently hash-pinned.

~~~sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export TORCH_PY=/home/amir/miniconda3/envs/mp_lstm/bin/python
export MATCH_PY=/absolute/path/to/pinned-matcher/bin/python
V=lattice_symmetry/v3
python "$V/protocol/lock.py"
python "$V/protocol/regress.py"
"$MATCH_PY" "$V/evaluation/score.py" "$V/protocol/historic-replay.npz" --split test --output "$V/protocol/historic-score.npz" --public
~~~

On the recorded Windows host the working matcher is `C:/Users/ASUS/AppData/Local/Temp/crystal-matcher-venv/Scripts/python.exe`. The default Windows Python was used for compact CPU inference. Set the above paths to the installations actually used on another machine. The split generator refuses to overwrite a different registered split. Reproduction from scratch belongs in a fresh extraction so the atomic checkpoints keep their original provenance.

## Registered inputs and choices

`protocol/PROTOCOL.md`, `protocol/analysis-plan.md`, `protocol/matrix.json` and `models/EXTENDED.md` describe training, finite-horizon choices and protected cases. `geometry/ARRANGEMENT-PROTOCOL.md` describes the outcome-blind secondary validation challenge that distinguishes genuinely unmatched arrangements from new formulas with known anonymous training analogues. `geometry/README.md` maps the mathematical statements to property tests, finite-difference fixtures and adversarial controls.

## From frozen networks to comparable candidates

The saved full-size external predictions, the earlier fixed-basis models and the supplied-template experiment remain distinct comparators. Every new test prediction shard stores its exact target IDs, numbered draw positions, cell, fractional positions, species and generator identity. Symbolic methods also save the training source of each pattern and cell. Scoring is separate and proposal-first. Blank or invalid proposals stay in the denominator. The raw predictions use native cells; reconstruction into a metric chart discards a global Cartesian rotation only. Console logs and temporary optimizer files are unnecessary for reducing completed runs.

~~~sh
python "$V/evaluation/register.py" first
python "$V/evaluation/register.py" baseline
python "$V/evaluation/controlled.py"
python "$V/evaluation/exposure.py"
python "$V/evaluation/paired.py" "$V/evaluation/first-frozen.json" --panel family --draws 20
"$MATCH_PY" "$V/evaluation/shards.py" family prior angular-small-11 contacts-small-11 reciprocal-small-11 angular-medium-11 --workers 2
python "$V/evaluation/analysis.py" family
python "$V/protocol/status.py"
~~~

Use the same paired driver with `--panel test`, `shared`, `arrangement` and `arrangement-control` for the declared public and protected panels. `evaluation/series.py` resumes a matrix across multiple panels. Dates, split salts, model selection checksums, input restrictions and bootstrap units remain inside the evidence. A training run uses the dev panel only to choose between its registered checkpoints.

## De novo and relaxation

The root formula source is an explicit empirical draw from the protected training compositions. The no-update, released stochastic angular, released uFlowCSP and symbolic-refinement arms can be regenerated on those same compositions. Their geometry seeds, templates and clock receipts remain arm-specific. The hash-selected CHGNet panel is determined before any energy is read. Raw novelty uses the protected training catalog. The relaxed conservative novelty check compares against all 27,136 official training records.

~~~sh
python "$V/denovo/pool.py" prior
python "$V/denovo/pool.py" contacts-small --checkpoint "$V/models/fits/small-contacts-laplace-11/full.pt"
python "$V/denovo/pool.py" released-angular --released
"$TORCH_PY" "$V/denovo/pool_uflow.py"
"$MATCH_PY" "$V/denovo/evaluate.py" contacts-small --workers 2
"$MATCH_PY" "$V/denovo/physical.py" prior contacts-small --workers 1
~~~

The final author-review guide and claims table identify the selected extended checkpoint, the complete set of measured methods and archived experimental failures. The old 38- and 21-page papers are separate releases. No external submission is part of the replay.
