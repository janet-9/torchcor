#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8
#$ -l h_vmem=11G
#$ -l h_rt=240:0:0
#$ -t 1-100
#$ -l gpu=1         # request 1 GPU


python ./atrium.py -case_id ${SGE_TASK_ID}