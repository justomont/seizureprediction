#!/bin/bash

#SBATCH --job-name=MonopolarTrainTestSeizPred
#SBATCH --array=0-5 # Include as many as methods
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-type=END
#SBATCH --mail-user=justo.montoya@upf.edu
#SBATCH -o outputs/offlineTrainTestSeizPred_%A_%a.out # Standard output
#SBATCH -e outputs/offlineTrainTestSeizPred_%A_%a.err # Standard error
DIR=/homes/users/jmontoya

python train.py $SLURM_ARRAY_TASK_ID 