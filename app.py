import streamlit as st
import whisper
import tempfile

# Configuration de la page
st.set_page_config(page_title="Transcription Audio", layout="centered")

st.title("🎙️ Application de Transcription Audio avec Whisper")

# Charger le modèle (base ou small/medium/large selon la puissance dispo)
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# Uploader fichier audio
uploaded_file = st.file_uploader(
    "Choisissez un fichier audio à transcrire",
    type=["mp3", "wav", "m4a"]
)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Sauvegarde temporaire du fichier
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_audio_path = tmp_file.name

    st.write("⏳ Transcription en cours...")

    # Transcription avec Whisper
    result = model.transcribe(temp_audio_path, fp16=False)

    st.success("✅ Transcription terminée !")
    st.subheader("Texte transcrit :")
    st.write(result["text"])
