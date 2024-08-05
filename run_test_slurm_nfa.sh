#!/bin/bash

#SBATCH --job-name=nemo_docker_asds_asr_job
#SBATCH --time=06:00:00 # Adjust based on your job's expected runtime
#SBATCH --gres=gpu:1 # Request 1 GPU, adjust as needed
#SBATCH --mem=16G # Adjust based on your job's memory requirements
#SBATCH --cpus-per-task=4 # Adjust based on your job's CPU requirements

python NeMo/tools/nemo_forced_aligner/align.py    model_path="/home/asds/ml_projects_mher/NeMo-SDP-ArmData/experiments/lang-hy-AM/ASR-Model-Language-hy-AM/2024-05-13_22-20-22/checkpoints/ASR-Model-Language-hy-AM.nemo"         manifest_filepath="/home/asds/ml_projects_mher/NeMo-SDP-ArmData/audio_durations_tmp.json"         output_dir="/home/asds/ml_projects_mher/NeMo-SDP-ArmData/tmp_output" align_using_pred_text=True

