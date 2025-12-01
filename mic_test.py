# file: mic_test.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av

# A class that will receive audio frames from your mic
class AudioProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        st.write("🎧 Received audio frame")
        return frame

# Streamlit app layout
st.title("🎙️ Microphone Test")

# Launch mic input using webrtc_streamer
webrtc_streamer(
    key="mic-test",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor,
)
