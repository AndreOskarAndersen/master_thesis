#!/bin/bash
#SBATCH --job-name=wpr684
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-10:00:00
#SBATCH --array 1-2%2

# Loading venv
module load python/3.9.9
python3 -m venv venv
source ./venv/activate

# Installing libraries
/home/wpr684/master_thesis/venv/bin/python3 -m pip --no-cache-dir install --upgrade pip
/home/wpr684/master_thesis/venv/bin/python3 -m pip --no-cache-dir install -r requirements.txt

# Preparing data
cd ./src/features
python3 -m preprocess_penn_action ${SLURM_ARRAY_TASK_ID}
