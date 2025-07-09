#!/bin/bash
#SBATCH --job-name=pacBipSDAVBM
#SBATCH --cpus-per-task=1
#SBATCH --array=0-1
#SBATCH --mem-per-cpu=5G
#SBATCH --mail-type=END
#SBATCH --mail-user=justo.montoya@upf.edu
#SBATCH -o outputs/connectivityAnalysis_%A_%a.out # Standard output
#SBATCH -e outputs/connectivityAnalysis_%A_%a.err # Standard error
DIR=/homes/users/jmontoya
python connectivity_analysis.py $SLURM_ARRAY_TASK_ID
