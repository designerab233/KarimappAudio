# audio_to_text_streamlit.py
# Copyright © 2025 Abdessamad Karim
# All rights reserved.

import whisper
import streamlit as st

st.set_page_config(page_title="Audio to Text by Karim", layout="wide")

st.title("🎤 Audio → Text Transcription")

# Upload audio file
audio_file = st.file_uploader("Select an audio file", type=["mp3", "wav", "m4a"])

# Load model (lazy load)
if "model" not in st.session_state:
    with st.spinner("Loading Whisper model (base)..."):
        st.session_state.model = whisper.load_model("base")

model = st.session_state.model

if audio_file is not None:
    st.audio(audio_file, format="audio/mp3")
    
    if st.button("Transcribe"):
        with st.spinner("Transcribing..."):
            # Save uploaded file temporarily
            with open("temp_audio", "wb") as f:
                f.write(audio_file.getbuffer())
            
            # Transcribe
            result = model.transcribe("temp_audio")
            transcription = result["text"]
            
            st.success("Transcription completed!")
            st.text_area("Transcription", transcription, height=300)
