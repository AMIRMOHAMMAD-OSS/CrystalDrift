# Reproducing the full-chemistry de novo experiment

Run the following from the project root after fitting the selected formula-conditioned checkpoints. The [design](DESIGN.md) fixes the chemical prior, draw order, geometry/novelty metrics and the 16-draw physical cohort. The new geometry prior and feature scales use the 27,136 training structures. The train-only chemical prior and train-only novelty catalog are shared with the earlier study. The 9,046 MP-20 test structures provide the coverage and property reference. The restricted competing-phase library uses all three MP-20 splits under the documented energy cutoff, with candidate and reference structures relaxed under the same potential.

The Torch environment generates the structures. The evaluator uses the pinned pymatgen/SMACT/matminer and CHGNet versions described in `expansion/denovo/README.md` and `expansion/README.md`. The lightweight compiled site-fingerprint extension is identical to the previous release. Specify an interpreter providing those dependencies; the paths below use arguments so they can be set locally.

~~~sh
python lattice_symmetry/v2/denovo/generate_pools.py fixed-basis
export CRYSTAL_METRICS_SITE=/absolute/path/to/metrics-site-packages
export MATCHER_PYTHON=/absolute/path/to/matcher/bin/python
for phase in raw fp novelty report; do
  "$MATCHER_PYTHON" lattice_symmetry/v2/denovo/evaluate.py "$phase" --arms fixed-basis --workers 1
done
python lattice_symmetry/v2/denovo/pipeline.py --matcher-python "$MATCHER_PYTHON" --metrics-site "$CRYSTAL_METRICS_SITE" --workers 1 --screen-workers 1
python lattice_symmetry/v2/denovo/energy_panel.py --matcher-python "$MATCHER_PYTHON" --metrics-site "$CRYSTAL_METRICS_SITE" --pyg-site /absolute/path/to/compatible-pyg
~~~

The optional `--wait` switch on either pipeline makes it wait for completed upstream selections or panels. Each pool contains the original draw IDs, chemistry, log lattice metric, native lattice, fractional sites, finite-output flags and batch durations. Its companion CPU JSON identifies the generator checkpoint, prior and software. Each new generator has a frozen copy of its arm-specific validation selection. Splitting work by stage or resuming completed tiles preserves draw order. Generation timers cover the physical-output boundary, and physical records charge the separate candidate, reference and final matching costs.

The coverage files retain distances to chemistry and structure cutoffs in both directions. Case files retain failure flags, uniqueness, training matches and the 1,000 valid indices used for property distribution measurements. The released supervised formation-energy predictor is a separate diagnostic evaluated on these preselected indices. Every property array records ID, index and eV/atom, with the model digest and all 9,046 reference predictions identified in the corresponding provenance record.
