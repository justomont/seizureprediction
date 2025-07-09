#!/bin/bash
#SBATCH --job-name=OnlinePredLap
#SBATCH --cpus-per-task=1
#SBATCH --array=0-30
#SBATCH --mem-per-cpu=5G
#SBATCH --mail-type=END
#SBATCH --mail-user=justo.montoya@upf.edu
#SBATCH -o outputs/onlinePrediction_%A_%a.out # Standard output
#SBATCH -e outputs/onlinePrediction_%A_%a.err # Standard error
DIR=/homes/users/jmontoya
python online_prediction.py $SLURM_ARRAY_TASK_ID