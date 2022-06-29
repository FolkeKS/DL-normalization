#!/bin/bash
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name train_net
#SBATCH --time=12:00:00
#SBATCH --output=results/outputs/slurm-%A.%a.out
#SBATCH --error=results/outputs/slurm-%A.%a.err
unset PYTHONHOME

   
