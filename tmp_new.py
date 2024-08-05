import json

# File paths
files_to_combine = [
    # '/workspace/nemo_capstone/train_manifest0.json',
    # '/workspace/nemo_capstone/test_manifest0.json',
    # '/workspace/nemo_capstone/validation_manifest0.json',
    '/workspace/nemo_capstone/data/mozilla-foundation/common_voice_16_1/hy-AM/other/other_mozilla-foundation_common_voice_16_1_manifest.json'
]

# The file to append the combined data to
target_file_path = '/workspace/nemo_capstone/data/mozilla-foundation/common_voice_16_1/hy-AM/train/train_mozilla-foundation_common_voice_16_1_manifest.json'

combined_json_objects = []

# Read and convert each file's contents to one-line JSON
for file_path in files_to_combine:
    with open(file_path, 'r') as file:
        for line in file:
            # Assuming each line is a valid JSON object
            json_obj = json.loads(line)
            combined_json_objects.append(json_obj)

# Adding the contents of the target file to the combined list
with open(target_file_path, 'r') as file:
    for line in file:
        json_obj = json.loads(line)
        combined_json_objects.append(json_obj)

# Write the combined one-line JSON objects back to the target file
with open(target_file_path, 'w') as file:
    for json_obj in combined_json_objects:
        # Convert each JSON object to a string and write it as a new line
        file.write(json.dumps(json_obj) + '\n')