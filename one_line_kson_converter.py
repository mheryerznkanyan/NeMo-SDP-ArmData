import json

def reformat_json_to_single_line(input_file_path, output_file_path):
    # Open the input file and read the JSON data
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)
    
    # Open the output file for writing
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for item in data:
            # Write each JSON object to a single line in the output file, preserving Unicode characters
            json.dump(item, output_file, ensure_ascii=False)
            output_file.write('\n')  # Add a newline character after each JSON object

# Specify the input and output file paths
input_file_path = '/workspace/nemo_capstone/train_manifest0.json'
output_file_path = '/workspace/nemo_capstone/data/mozilla-foundation/common_voice_16_1/hy-AM/train/train_mozilla-foundation_common_voice_16_1_manifest.json'

# Call the function with the specified paths
reformat_json_to_single_line(input_file_path, output_file_path)

print("JSON data has been reformatted to single-line objects in the output file.")
