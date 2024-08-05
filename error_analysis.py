import json
import nemo.collections.asr as nemo_asr

# Load your pre-trained ASR model
asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained('mheryerznka/stt_arm_fastconformer_hybrid_large_pc', map_location='cpu') 

# model.change_decoding_strategy(decoder_type='ctc')

# Path to your existing manifest file
manifest_path = '/home/asds/ml_projects_mher/NeMo-SDP-ArmData/asds/mozilla-foundation_copy_fleurs_mcv_copy/common_voice_17_0/hy-AM/test/test_mozilla-foundation_common_voice_17_0_manifest.json.clean'

# Prepare to write to the updated manifest file
output_path = '/home/asds/ml_projects_mher/NeMo-SDP-ArmData/asds/mozilla-foundation_copy_fleurs_mcv_copy/common_voice_17_0/hy-AM/test/test_mozilla-foundation_common_voice_17_0_manifest_slimipl_2024-06-25_21-58-58.json'

# Process each line in the original manifest
with open(manifest_path, 'r') as infile, open(output_path, 'w') as outfile:
    # Reading and collecting batch data
    audio_file_paths = []
    entries = []
    # breaking = 0
    for line in infile:
        entry = json.loads(line)
        entries.append(entry)
        audio_file_paths.append(entry['audio_filepath'])
        # breaking+=1

    # Transcribe all collected audio files in a single batch
    pred_texts = model.transcribe(audio_file_paths)

    # Write updated entries to new file
    for entry, pred_text in zip(entries, pred_texts[0]):
        entry['pred_text'] = pred_text
        json.dump(entry, outfile, ensure_ascii=False)  # Ensure non-ASCII characters are saved properly
        outfile.write('\n')

print("Updated manifest saved with predicted transcriptions.")