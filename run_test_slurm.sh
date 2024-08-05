#!/bin/bash

#SBATCH --job-name=nemo_docker_asds_asr_job
#SBATCH --time=06:00:00 # Adjust based on your job's expected runtime
#SBATCH --gres=gpu:1 # Request 1 GPU, adjust as needed
#SBATCH --mem=16G # Adjust based on your job's memory requirements
#SBATCH --cpus-per-task=4 # Adjust based on your job's CPU requirements

python /home/asds/ml_projects_mher/NeMo-SDP-ArmData/test_my_code_v4.py

