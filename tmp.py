import nemo.collections.asr as nemo_asr

# Load the pre-trained ASR model
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained("mheryerznka/stt_arm_fastconformer_hybrid_large_pc",map_location='cpu')

# Confirm the model is on the CPU
print("Model is on device:", asr_model.device)

# Path to the WAV file
audio_file_path = 'hy_am-test-26-audio-audio.wav'

# Transcribe the audio file
transcription = asr_model.transcribe([audio_file_path])

# Print the transcription result
print("Transcription:", transcription)