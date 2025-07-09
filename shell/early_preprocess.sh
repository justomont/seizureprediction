#!/bin/bash
#SBATCH --job-name=pPBipVBMSDA
#SBATCH --array=0-30 # Include as many as files in RAW data
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --mail-type=END
#SBATCH --mail-user=justo.montoya@upf.edu
#SBATCH -o outputs/preprocess_%A_%a.out # Standard output
#SBATCH -e outputs/preprocess_%A_%a.err # Standard error
DIR=/homes/users/jmontoya
# For the next step, choose the module Python/X.X.X that you need (module av to see available modules):
# module load Python/3.9.12 
# conda activate testenv
python early_preprocess.py $SLURM_ARRAY_TASK_ID