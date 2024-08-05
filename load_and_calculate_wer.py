import nemo.collections.asr as nemo_asr
import json
from nemo.collections.asr.metrics.wer import word_error_rate

# Load your pre-trained model (assuming it's a .nemo file)
model = nemo_asr.models.EncDecCTCModel.restore_from('experiments/lang-hy-AM/ASR-Model-Language-hy-AM/2024-04-17_20-45-50/checkpoints/ASR-Model-Language-hy-AM.nemo')



# Load the manifest
test_manifest_path = 'asds/mozilla-foundation/common_voice_16_1/hy-AM/test/test_mozilla-foundation_common_voice_16_1_manifest.json.clean'
with open(test_manifest_path, 'r') as f:
    test_manifest = [json.loads(line) for line in f]

transcriptions = [entry['text'] for entry in test_manifest]
audio_file_paths = [entry['audio_filepath'] for entry in test_manifest]  # List of audio file paths

# Transcribe all audio files in a single batch
hypotheses = model.transcribe(audio_file_paths)


# print(hypotheses[0])
print("hypotheses", len(hypotheses[0]))
print("transcriptions", len(transcriptions))
# Assert the lengths of hypotheses and transcriptions are the same
assert len(hypotheses[0]) == len(transcriptions), "Length of hypotheses and transcriptions must match"

# Calculate WER
wer = word_error_rate(hypotheses[0], transcriptions)
print(f"Word Error Rate: {wer:.2f}%")