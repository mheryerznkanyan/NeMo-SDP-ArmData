import json
import mutagen.mp3

# Path to your JSON file
json_file_path = "/workspace/nemo_capstone/data/mozilla-foundation/common_voice_16_1/hy-AM/train/train_mozilla-foundation_common_voice_16_1_manifest.json"

# Function to calculate audio duration
def get_audio_duration(audio_file_path):
    audio = mutagen.mp3.MP3(audio_file_path)
    return audio.info.length

# Initialize total duration
total_duration = 0

# Read JSON file and process each line
# Initialize total duration
total_duration = 0

# Read JSON file and process each line
with open(json_file_path, 'r') as file:
    for line in file:
        # Parse JSON
        data = json.loads(line)

        # Extract duration from JSON data
        duration_seconds = data.get('duration', 0)

        # Add duration to total
        total_duration += duration_seconds

# Print total duration
print("Total duration of all audio files:", total_duration/60, "minutes")