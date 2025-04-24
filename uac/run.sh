#!/bin/bash

#$ -cwd
#$ -j y
#$ -l rocky
#$ -pe smp 8        # Request cores (8 per GPU)
#$ -l h_vmem=7.5G   # Request RAM (7.5GB per core)
#$ -l h_rt=1:0:0    # Max 1hr runtime (can request up to 240hr)
#$ -l gpu=1         # Request GPU
#$ -l gpu_type=hopper  # Ensure that the job runs on Andrena nodes
#$ -N uac     # Name for the job (optional)

# Load the necessary modules
module load miniforge
module load cudnn/8.9.7.29-12-cuda-12.4.0-gcc-12.2.0
conda activate venv

python ./train.py -root "/data/scratch/acw554"