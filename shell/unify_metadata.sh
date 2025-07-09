#!/bin/bash
#SBATCH --job-name=unifymetadata
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END
#SBATCH --mail-user=justo.montoya@upf.edu
DIR=/homes/users/jmontoya
python unify_metadata.py 
