import json
import os
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment

# Initialize the Forced Aligner
asr_model = nemo_asr.models.ASRModel.restore_from('/home/asds/ml_projects_mher/NeMo-SDP-ArmData/experiments/lang-hy-AM/ASR-Model-Language-hy-AM/2024-05-13_22-20-22/checkpoints/ASR-Model-Language-hy-AM.nemo',  map_location='cpu')

# Load the JSON data with filtered audio files
with open('audio_durations_filtered.json', 'r') as file:
    data = json.load(file)

# Function to align audio files using NeMo Forced Aligner
def align_audio_files(data, asr_model):
    aligned_data = []
    for entry in data:
        audio_path = entry['audio_filepath']
        duration = entry['duration']
        
        # Perform forced alignment to get word timings
        transcription, word_timings = asr_model.transcribe([audio_path], return_word_timestamps=True)
        
        # Add timings to JSON entry
        entry['transcription'] = transcription[0]
        entry['word_timings'] = word_timings[0]
        aligned_data.append(entry)
    
    return aligned_data

# Align the audio files and get word timings
aligned_data = align_audio_files(data, asr_model)

# Save the aligned data to a new JSON file
with open('aligned_data.json', 'w') as file:
    json.dump(aligned_data, file, indent=4)

# Print the aligned data
# print(json.dumps(aligned_data, indent=4))