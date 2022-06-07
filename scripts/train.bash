#!/bin/bash
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name train_unet
#SBATCH --time=12:00:00
#SBATCH --output=results/outputs/slurm-%A.%a.out
unset PYTHONHOME

srun python scripts/trainer.py fit --config configs/kraken_demo.yaml> results/outputs/${SLURM_JOBID}.out
   
