#!/bin/bash

#SBATCH -p express
#SBATCH -c 1
#SBATCH -t 10
#SBATCH --array=0-9

module load anaconda
conda env list
conda activate plotly
export SEED=1
python epidemic_expt.py $SEED $SLURM_ARRAY_TASK_ID
