#!/bin/bash
#SBATCH --job-name=multielement-crystals
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=multielement-%j.out
#SBATCH --error=multielement-%j.err
set -euo pipefail

# Pre-stage the processed data on shared storage. Submit from scaled/.
# sbatch --account=PROJECT@h100 jean_zay_scaled.slurm wbm
target=${1:-mp20}
case "$target" in mp20|wbm|alex-training-view) ;; *) echo 'Choose mp20, wbm or alex-training-view.' >&2; exit 2 ;; esac
source_dir=${SLURM_SUBMIT_DIR:?Submit from scaled/}
result_root=${CRYSTAL_RUN_ROOT:-"$source_dir/job-results"}
mkdir -p -- "$result_root"
run="$result_root/${SLURM_JOB_ID:?Missing job ID}"
mkdir -- "$run"
cd -- "$run"
cp -- "$source_dir"/{family,experiment,evaluate,summarize,check,prepare_mp20,audit,latency}.py .
cp -- "$source_dir/jean_zay_scaled.slurm" .
mkdir "$target-data"
cp -- "$source_dir/$target-data/"{train.npz,val.npz,test.npz,manifest.json} "$target-data/"
sha256sum ./*.py ./*.slurm "$target-data/"*.npz > inputs.sha256
module purge
module load arch/h100
module load "${CRYSTAL_TORCH_MODULE:-pytorch-gpu/py3/2.5.0}"
module list > modules.txt 2>&1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
scontrol show job "$SLURM_JOB_ID" > slurm-job.txt
srun nvidia-smi > gpu.txt
srun python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name())' > cuda.txt
srun python check.py
srun python experiment.py --data "$target-data" --out "runs/$target" --device cuda --epochs 1 --threads 2 --seeds 11 23 37
srun python audit.py --train "$target-data" --data "$target-data" --out "runs/$target"
srun python evaluate.py --data "$target-data" --out "runs/$target" --seeds 11 23 37
srun python summarize.py "runs/$target" --seeds 11 23 37
srun python latency.py --data "$target-data" --run "runs/$target" --export --device cpu --seeds 11 23 37
srun python latency.py --data "$target-data" --run "runs/$target" --device cuda --seeds 11 23 37
test -s "runs/$target/geometry-summary.json"
printf 'Job %s completed. Results and checkpoints: %s/runs/%s/\n' "$SLURM_JOB_ID" "$PWD" "$target"
