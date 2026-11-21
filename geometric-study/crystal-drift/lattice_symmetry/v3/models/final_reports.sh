#!/usr/bin/env bash
# Set the current training interpreter in PATH. Run from the release root.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
py="${CRYTALDRIFT_PYTHON:-python}"
"$py" -m lattice_symmetry.v3.models.scale_report
"$py" -m lattice_symmetry.v3.models.report_validation
"$py" -m lattice_symmetry.v3.models.capacity_paired
"$py" -m lattice_symmetry.v3.models.diagnose
"$py" -m lattice_symmetry.v3.models.replications
"$py" -m lattice_symmetry.v3.models.learning_curves
"$py" -m lattice_symmetry.v3.models.fixed_cost
"$py" -m lattice_symmetry.v3.models.training_order
"$py" -m lattice_symmetry.v3.models.verify_fits --complete
"$py" -m lattice_symmetry.v3.models.audit_terminal
"$py" -m lattice_symmetry.v3.models.weights_manifest
"$py" -m lattice_symmetry.v3.models.weights_manifest --selected
