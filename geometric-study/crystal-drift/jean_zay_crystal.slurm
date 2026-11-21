#!/bin/bash
#SBATCH --job-name=crystal-drift
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:45:00
#SBATCH --output=crystal-%j.out
#SBATCH --error=crystal-%j.err
set -euo pipefail

# From crystal-drift on Jean Zay, replace PROJECT with your allocation name.
#   sbatch --account=PROJECT@h100 jean_zay_crystal.slurm
# Append `retrain` to refit all four compact models on CPU for three seeds.
# IDRIS module example: https://github.com/idriscnrs/tutorials_jean-zay_picotron
mode=${1:-bundled}
case "$mode" in
    bundled|retrain) ;;
    *) echo 'Choose bundled or retrain.' >&2; exit 2 ;;
esac

study=${SLURM_SUBMIT_DIR:?Submit from the crystal-drift directory}
run_root=${CRYSTAL_RUN_ROOT:-"$study/job-results"}
mkdir -p -- "$run_root"
run="$run_root/${SLURM_JOB_ID:?Missing Slurm job ID}"
mkdir -- "$run"
cd -- "$run"
printf 'Job %s, mode %s, output %s\n' "$SLURM_JOB_ID" "$mode" "$PWD"

# Each Python entry point locates its output relative to its own file. Staging
# a copy here also preserves the precise study code and inputs for this run.
cp -- "$study"/{baseline.py,baseline_multi.py,model.py,speed.py,train.py,jean_zay_crystal.slurm} .
mkdir -p data speed-results/portable
cp -- "$study"/data/{train,test,multi-train,multi-test}.npz data/
sha256sum ./*.py ./*.slurm data/*.npz > inputs.sha256

module purge
module load arch/h100
module load "${CRYSTAL_TORCH_MODULE:-pytorch-gpu/py3/2.5.0}"
module list > modules.txt 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
scontrol show job "$SLURM_JOB_ID" > slurm-job.txt
srun nvidia-smi > gpu.txt
srun python -c 'import numpy,torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(),numpy.__version__)' > runtime.txt

if [[ $mode == retrain ]]; then
    srun python train.py --methods ab-exact multi-exact --seeds 11 23 37
    srun python baseline.py --seeds 11 23 37
    srun python baseline_multi.py --seeds 11 23 37
    srun python speed.py --export --device cpu --repeats 15
else
    for cohort in ab multi; do
        for method in drift flow; do
            for seed in 11 23 37; do
                cp -- "$study/speed-results/portable/$cohort-$method-$seed.npz" speed-results/portable/
            done
        done
    done
    srun python speed.py --device cpu --repeats 15
fi
sha256sum speed-results/portable/*.npz > checkpoints.sha256
srun python speed.py --device cuda --repeats 15

for device in cpu cuda; do
    test -s "speed-results/timings-$device.csv"
    test -s "speed-results/timings-$device.json"
done
printf 'Completed job %s. Timing tables and hardware metadata are in %s/speed-results/\n' "$SLURM_JOB_ID" "$PWD"
