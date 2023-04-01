#!/bin/bash
#SBATCH -p gpu --gres=gpu:titanrtx:2
#SBATCH --job-name=wpr684
#SBATCH --cpus-per-task=4
#SBATCH --mem 8000000K
#SBATCH --time=4-02:00:00
#SBATCH --array 0-1%2

# Loading modules
module load python/3.9.9
module load cuda/11.8

export LD_LIBRARY_PATH=/opt/software/cuda/11.8/lib64:$LD_LIBRARY_PATH
export PATH=/opt/software/cuda/11.8/bin:$PATH

# Changing directory to master_thesis
cd master_thesis

# Loading virtual environment
python3 -m venv venv
source ./venv/bin/activate

# Installing libraries
/home/wpr684/master_thesis/venv/bin/python3 -m pip --no-cache-dir install --upgrade pip
/home/wpr684/master_thesis/venv/bin/python3 -m pip --no-cache-dir install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
/home/wpr684/master_thesis/venv/bin/python3 -m pip --no-cache-dir install -r requirements.txt

hostname
echo $CUDA_VISIBLE_DEVICES

# Running script
cd ./src/models
python3 -m continue_training ${SLURM_ARRAY_TASK_ID}
