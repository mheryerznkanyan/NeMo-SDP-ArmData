import nemo.collections.asr as nemo_asr
import json
import os
import soundfile as sf


def convert_to_mono(audio_path):
    data, samplerate = sf.read(audio_path)
    if len(data.shape) > 1:  # Check if audio is stereo
        data = data.mean(axis=1)  # Convert to mono
        sf.write(audio_path, data, samplerate)

model_path = "/home/asds/ml_projects_mher/NeMo-SDP-ArmData/experiments/lang-hy-AM/ASR-Model-Language-hy-AM/2024-05-13_22-20-22/checkpoints/ASR-Model-Language-hy-AM.nemo"
asr_model = nemo_asr.models.ASRModel.restore_from(model_path,  map_location='cpu')

audio_directory = "/home/asds/ml_projects_mher/NeMo-SDP-ArmData/asds/test_files"
audio_extensions = {".mp3"}

audio_files = [os.path.join(audio_directory, f) for f in os.listdir(audio_directory) if os.path.splitext(f)[1] in audio_extensions]

transcriptions = []
for audio_file in audio_files:
    convert_to_mono(audio_file)
    transcription = asr_model.transcribe([audio_file])
    transcriptions.append({"audio_filepath": audio_file, "text": transcription[0]})

manifest_path = "yervand_test_manifest.json"
with open(manifest_path, 'w') as manifest_file:
    for transcription in transcriptions:
        json.dump(transcription, manifest_file, ensure_ascii=False)
        manifest_file.write('\n')

print(f"Manifest file saved to {manifest_path}")