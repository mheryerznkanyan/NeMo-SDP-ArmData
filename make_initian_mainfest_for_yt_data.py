import os
import wave
import json

def get_wav_duration(wav_path):
    with wave.open(wav_path, 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
    return duration

def write_durations_to_jsonl(root_dir, output_file):
    # Create or overwrite the JSONL file
    with open(output_file, 'w') as file:
        # Walk through the directory structure
        for root, dirs, files in os.walk(root_dir):
            for filename in files:
                if filename.endswith('.wav'):
                    filepath = os.path.join(root, filename)
                    duration = get_wav_duration(filepath)
                    # Create a dictionary with the required data
                    data = {"audio_filepath": filepath, "duration": duration}
                    # Write the data as a JSON line
                    file.write(json.dumps(data) + '\n')

# Specify the root directory containing WAV files
root_directory = '/home/asds/ml_projects_mher/NeMo-SDP-ArmData/wavs'
# Specify the output JSON Lines file
output_jsonl_file = 'audio_durations_v2.json'

# Call the function to process the files
write_durations_to_jsonl(root_directory, output_jsonl_file)

print("Duration data has been written to", output_jsonl_file)
