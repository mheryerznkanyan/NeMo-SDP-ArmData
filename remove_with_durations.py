import json

# Load the JSON data
json_path = '/home/asds/ml_projects_mher/NeMo-SDP-ArmData/audio_durations_v2.json'
with open(json_path, 'r') as file:
    data = [json.loads(line) for line in file]

# Filter out entries with duration more than 20 seconds
filtered_data = [entry for entry in data if entry.get('duration', 0) <= 20 and entry.get('duration', 0) >= 1.5]

# Save the filtered data back to the JSON file
filtered_json_path = '/home/asds/ml_projects_mher/NeMo-SDP-ArmData/initmanifest0.json'
with open(filtered_json_path, 'w') as file:
    for entry in filtered_data:
        file.write(json.dumps(entry) + '\n')