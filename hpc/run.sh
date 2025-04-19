#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8
#$ -l h_vmem=7.5G
#$ -l h_rt=240:0:0
#$ -t 1-100
#$ -l gpu=1
#$ -N conductivity

module load miniforge
module load cudnn/8.9.7.29-12-cuda-12.4.0-gcc-12.2.0
conda activate venv


python ./atrium_cm.py -case_id ${SGE_TASK_ID}