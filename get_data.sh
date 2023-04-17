#!/bin/bash
#SBATCH --job-name=wpr684
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00

# Loading venv
module load python/3.9.9
python3 -m venv venv
source ./venv/activate

# Installing libraries
/home/wpr684/master_thesis/venv/bin/python3 -m pip --no-cache-dir install --upgrade pip
/home/wpr684/master_thesis/venv/bin/python3 -m pip --no-cache-dir install -r requirements.txt

# Downloading data
cd ./src/data
python3 -m make_dataset

# Preparing data
cd ../features
python3 -m build_features