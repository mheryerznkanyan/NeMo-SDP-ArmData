import json

def load_json_lines(file_path):
    """Load JSON objects from a file formatted with JSON Lines, skipping empty lines."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            try:
                if line.strip():  # Check if the line is not just whitespace
                    data.append(json.loads(line))
                else:
                    print(f"Skipping empty line in {file_path} at line {line_number}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line in {file_path} at line {line_number}: {e}")
    return data

def save_json(data, file_path):
    """Save JSON data to a file, ensuring Unicode characters are correctly saved."""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def main():
    # Paths to input JSON files
    file1_path = '/workspace/nemo_capstone/data/mozilla-foundation/common_voice_16_1/hy-AM/other/other_mozilla-foundation_common_voice_16_1_manifest.json'
    file2_path = '/workspace/nemo_capstone/data/mozilla-foundation/common_voice_16_1/hy-AM/train/train_mozilla-foundation_common_voice_16_1_manifest.json'

    # Load data from both files
    data1 = load_json_lines(file1_path)
    data2 = load_json_lines(file2_path)

    # Combine the data from both files into a single list
    combined_data = data1 + data2

    # Path to the combined JSON file
    combined_file_path = '/workspace/nemo_capstone/data/mozilla-foundation/common_voice_16_1/hy-AM/train/train_mozilla-foundation_common_voice_16_1_manifest.json'

    # Save the combined data
    save_json(combined_data, combined_file_path)

    print(f"Combined JSON saved to {combined_file_path}")

if __name__ == "__main__":
    main()