#!/bin/bash

#SBATCH --job-name=nemo_docker_asds_asr_job
#SBATCH --time=02:00:00 # Adjust based on your job's expected runtime
#SBATCH --gres=gpu:1 # Request 1 GPU, adjust as needed
#SBATCH --mem=32G # Adjust based on your job's memory requirements
#SBATCH --cpus-per-task=4 # Adjust based on your job's CPU requirements

docker exec -u root c3d6e1657af9 /bin/bash -c "/workspace/nemo_capstone/training_pipeline.sh"

