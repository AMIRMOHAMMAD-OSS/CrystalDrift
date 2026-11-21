# Running a controlled fit on Jean Zay

Restore the evidence and its named base archives into a fresh study tree. Run from the directory containing `data.py` and `lattice_symmetry/v3`. The preparation and training entry points read the protected split and the bank calibration from the research package. The study used one laptop GPU. No Jean Zay login, running job or allocation was configured in the study environment.

These commands take your existing A100 Slurm account and prepared PyTorch interpreter as inputs. The GPU interpreter must provide PyTorch with CUDA, NumPy, SciPy and Numba. Use the pinned matcher separately for evaluation. Run from a fresh working copy to reproduce training from the beginning. Existing completed checkpoints cause the trainer to resume or skip that fit.

```bash
read -rp "Jean Zay Slurm account including @a100: " CRYTALDRIFT_ACCOUNT
read -rp "Absolute path to CUDA Python on the compute node: " CRYTALDRIFT_PYTHON
export CRYTALDRIFT_PYTHON
sbatch --account="$CRYTALDRIFT_ACCOUNT" lattice_symmetry/v3/protocol/jean_zay.slurm --tier reference --arm contacts --affinity laplace --seed 11 --batch 2 --profile
sbatch --account="$CRYTALDRIFT_ACCOUNT" lattice_symmetry/v3/protocol/jean_zay.slurm --tier reference --arm contacts --affinity laplace --seed 11 --batch 2 --passes 1
squeue --me
```

The training and profile use different output directories. The profile measures throughput and peak tensor allocation. A completed full traversal records anchor exposures and produces weights and development predictions for frozen checkpoint selection. The project uses a single-device optimizer. A change in microbatch size also changes the number of updates, so keep that setting when reproducing a registered comparison. On Apple silicon, CPU inference and the evaluation code are separable from the CUDA trainer. This study did not profile an Apple GPU. [IDRIS Slurm examples and architecture modules](https://www.idris.fr/media/eng/ia/guide_nouvel_utilisateur_ia-eng.pdf)

