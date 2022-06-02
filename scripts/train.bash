#!/bin/bash
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name train_unet
#SBATCH --time=12:00:00
unset PYTHONHOME

srun python python scripts/trainer.py fit --config configs/kraken_learning_curve.yaml> results/outputs/${SLURM_JOBID}.out
   
