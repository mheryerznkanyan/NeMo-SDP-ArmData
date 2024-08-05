import json
import os
from pydub import AudioSegment

# Directory containing the .ctm files
ctm_directory = '/home/asds/ml_projects_mher/NeMo-SDP-ArmData/tmp_output/ctm/words'

# Directory to save the segmented audio files
output_directory = '/home/asds/ml_projects_mher/NeMo-SDP-ArmData/segmented_audio'
os.makedirs(output_directory, exist_ok=True)

# Path to the JSON file with audio paths
audio_paths_json = '/home/asds/ml_projects_mher/NeMo-SDP-ArmData/tmp_output/audio_durations_filtered_with_output_file_paths.json'

# Function to parse .ctm files
def parse_ctm(ctm_file):
    word_timings = []
    with open(ctm_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 5:
                start_time = float(parts[2])
                duration = float(parts[3])
                word = parts[4]
                end_time = start_time + duration
                word_timings.append({
                    "word": word,
                    "start_time": start_time,
                    "end_time": end_time
                })
    return word_timings

# Function to split audio files based on word timings
def split_audio_files(audio_path, word_timings, output_dir, max_duration=15):
    print(f"Processing audio file: {audio_path}")
    audio = AudioSegment.from_wav(audio_path)
    segmented_data = []
    max_duration_ms = max_duration * 1000  # 20 seconds in milliseconds

    current_segment = []
    current_duration = 0
    segment_start_time = 0
    segment_count = 0
    base_filename = os.path.basename(audio_path).split('.')[0]
    
    for word in word_timings:
        word_start = word['start_time'] * 1000  # convert to milliseconds
        word_end = word['end_time'] * 1000  # convert to milliseconds
        word_duration = word_end - word_start

        if current_duration + word_duration > max_duration_ms:
            # Save the current segment
            segment_audio = audio[segment_start_time:current_duration]
            segment_filename = f"{base_filename}_part{segment_count}.wav"
            segment_path = os.path.join(output_dir, segment_filename)
            segment_audio.export(segment_path, format="wav")

            segment_duration = len(segment_audio) / 1000.0
            print(f"Created segment: {segment_path} with duration {segment_duration} seconds")

            segmented_data.append({
                "audio_filepath": segment_path,
                "duration": segment_duration,  # duration in seconds
                "transcription": ' '.join([w['word'] for w in current_segment]),
                "word_timings": current_segment
            })

            # Start a new segment
            segment_start_time = word_start
            current_segment = [word]
            current_duration = word_duration
            segment_count += 1

            print(f"Starting new segment at word: {word['word']}, start_time: {word_start / 1000.0}, duration: {word_duration / 1000.0}")
        else:
            current_segment.append(word)
            current_duration += word_duration

    # # Save the last segment if any
    if current_segment:
        segment_audio = audio[segment_start_time:]
        segment_filename = f"{base_filename}_part{segment_count}.wav"
        segment_path = os.path.join(output_dir, segment_filename)
        segment_audio.export(segment_path, format="wav")

        segment_duration = len(segment_audio) / 1000.0
        print(f"Created segment: {segment_path} with duration {segment_duration} seconds")

        segmented_data.append({
            "audio_filepath": segment_path,
            "duration": segment_duration,  # duration in seconds
            "transcription": ' '.join([w['word'] for w in current_segment]),
            "word_timings": current_segment
        })
    
    return segmented_data

# Load the JSON data with audio paths (line-by-line)
all_segmented_data = []

print("Loading JSON data with audio paths...")
with open(audio_paths_json, 'r') as file:
    for line in file:
        entry = json.loads(line.strip())
        audio_path = entry['audio_filepath']
        audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
        ctm_path = os.path.join(ctm_directory, f"{audio_basename}.ctm")
        
        print(f"Checking existence of {ctm_path}")
        if os.path.exists(ctm_path):
            print(f"Found .ctm file: {ctm_path}")
            word_timings = parse_ctm(ctm_path)
            print(f"Word timings for {audio_basename}: {word_timings}")
            segmented_data = split_audio_files(audio_path, word_timings, output_directory)
            all_segmented_data.extend(segmented_data)
        else:
            print(f".ctm file not found: {ctm_path}")

# Save the segmented data to a new JSON file
output_json_path = 'segmented_data.json'
print(f"Saving segmented data to {output_json_path}...")
with open(output_json_path, 'w') as file:
    json.dump(all_segmented_data, file, indent=4)

# Print the segmented data
# print("Segmented data:")
# print(json.dumps(all_segmented_data, indent=4))

# Check the durations of all segmented audio files
exceeding_segments = []
print("Checking the durations of all segmented audio files...")
for segment in all_segmented_data:
    if segment['duration'] > 20:
        exceeding_segments.append(segment)

if exceeding_segments:
    print("Segments exceeding 20 seconds:")
    for segment in exceeding_segments:
        print(segment)
else:
    print("All segments are within the 20-second limit.")
