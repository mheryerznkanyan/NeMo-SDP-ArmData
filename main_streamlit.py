# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
# import numpy as np
# import soundfile as sf
# import tempfile
# import nemo.collections.asr as nemo_asr

# # Define a class to handle audio processing
# class AudioProcessor(VideoTransformerBase):
#     sample_rate = 16000
    
#     def __init__(self) -> None:
#         self.audio_buffer = []
#         self.asr_model = nemo_asr.models.ASRModel.restore_from("experiments/lang-hy-AM/ASR-Model-Language-hy-AM/2024-05-03_11-10-10/checkpoints/ASR-Model-Language-hy-AM.nemo", map_location='cpu')

#     def recv(self, frame):
#         data = frame.to_ndarray(format="s16")
#         self.audio_buffer.extend(data)
#         return frame

#     def process_audio(self):
#         # Process and clear buffer
#         if self.audio_buffer:
#             # Convert buffer to an appropriate format for your ASR model
#             with tempfile.NamedTemporaryFile(delete=True, suffix='.wav') as tmpfile:
#                 sf.write(tmpfile, np.array(self.audio_buffer), self.sample_rate)
#                 tmpfile.flush()
#                 # Perform transcription
#                 transcription = self.asr_model.transcribe(tmpfile.name, batch_size=1)
#             self.audio_buffer = []
#             return transcription[0] if transcription else "No transcription detected"

# def main():
#     st.title("Armenian ASR Testing App")

#     # WebRTC configuration for external usage, replace with your own STUN/TURN servers if needed
#     rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]} )

#     webrtc_ctx = webrtc_streamer(key="example", 
#                                  video_processor_factory=AudioProcessor,
#                                  rtc_configuration=rtc_configuration,
#                                  media_stream_constraints={"video": False, "audio": True})

#     # if webrtc_ctx.state.playing:
#     #     st.write("Recording...")
#     # else:
#     #     st.write("Click start to record")

#     if st.button("Stop Recording"):
#         webrtc_ctx.stop()

#     if st.button('Transcribe'):
#         transcription = webrtc_ctx.video_processor.process_audio() if webrtc_ctx.video_processor else "Start recording first"
#         st.write('Transcription:', transcription)

# if __name__ == "__main__":
#     main()


import streamlit as st
from audio_recorder_streamlit import audio_recorder
import soundfile as sf
import io
import nemo.collections.asr as nemo_asr
import librosa


# Load your NeMo ASR model
# model_path = 'path_to_your_model.nemo'
asr_model = nemo_asr.models.EncDecCTCModel.restore_from("/home/asds/ml_projects_mher/NeMo-SDP-ArmData/experiments/lang-hy-AM/ASR-Model-Language-hy-AM/2024-05-19_11-26-26/checkpoints/ASR-Model-Language-hy-AM.nemo", map_location='cpu')

# Title of the app
# Title of the app
st.title('Audio Recorder and Transcriber')

# Audio recorder component
audio_bytes = audio_recorder()
if audio_bytes:
    # Convert bytes data to audio array
    audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype='float32')

    # Check if the audio is stereo and convert to mono by averaging the two channels
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    # Expected sample rate from the model configuration
    target_sample_rate = asr_model.preprocessor._cfg['sample_rate']

    # Resample if the sample rates do not match
    if sample_rate != target_sample_rate:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sample_rate)
        st.write(f"Resampled from {sample_rate} Hz to {target_sample_rate} Hz.")

    # Prepare audio data for the model
    audio_data = audio_data[None, :]  # Add a batch dimension

    # change_in_dBFS = -20.0 - audio_data.dBFS
    # audio = audio_data.apply_gain(change_in_dBFS)

    # Transcribe the audio
    transcript = asr_model.transcribe(audio_data[0])[0]
    st.audio(audio_bytes, format="audio/wav")
    st.write(f"Transcription: {transcript}")
    st.success('Audio recorded and transcribed successfully!')
