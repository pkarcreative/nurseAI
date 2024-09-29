import streamlit as st
from PIL import Image
import sounddevice as sd
import numpy as np
import io
from scipy.io.wavfile import write
import wave
import tempfile
import os
from face_emotion_recognition import facial_emotion_detection, suggest_action_facial
from speech_emotion_recognition import voice_emotion_detection, suggest_action_voice

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center;'>NurseAI: Emotion Detection for Enhanced Patient Support</h1>", unsafe_allow_html=True)
st.write("<br>" * 2, unsafe_allow_html=True)

def capture_audio(filename, duration, fs=44100):
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")

    # Save the recorded audio
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 2 bytes per sample
        wf.setframerate(fs)
        wf.writeframes(audio_data.tobytes())
    return filename





# Split the page into two columns
col1, col2 = st.columns(2)



# Facial Emotion Detection in Column 1
with col1:
    st.header("Facial Emotion Detection")

    # Option to upload an image or capture from webcam
    image_input_type = st.radio("Choose input method:", ("Upload Image", "Capture Image"))

    if image_input_type == "Upload Image":
        # Upload image option
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)


            # Perform facial emotion detection
            result_facial = facial_emotion_detection(image)
            suggestion_facial = suggest_action_facial(result_facial)
            st.write(f"Detected Emotion: {result_facial.upper()}")
            st.write(f"Suggested Action: {suggestion_facial}")

    elif image_input_type == "Capture Image":
        # Capture image from webcam
        captured_image = st.camera_input("Capture an image")
        if captured_image is not None:
            image = Image.open(captured_image)
            #st.image(image, caption="Captured Image", use_column_width=True)

            # Perform facial emotion detection
            result_facial = facial_emotion_detection(image)
            suggestion_facial = suggest_action_facial(result_facial)
            st.write(f"Detected Emotion: {result_facial.upper()}")
            st.write(f"Suggested Action: {suggestion_facial}")

# Voice Emotion Detection in Column 2
with col2:
    st.header("Voice Emotion Detection")

    # Option to upload an audio file or record audio
    audio_input_type = st.radio("Choose input method:", ("Upload Audio"))

    if audio_input_type == "Upload Audio":
        # Upload audio option
        uploaded_audio = st.file_uploader("Upload an audio file", type=["wav"])
        if uploaded_audio is not None:
            # Play the uploaded audio
            st.audio(uploaded_audio, format="audio/wav")

            # Use a temporary file to store the uploaded audio and pass its path to the function
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                temp_audio_file.write(uploaded_audio.read())
                temp_audio_path = temp_audio_file.name
            # Perform voice emotion detection using the temporary file path
            result_voice = voice_emotion_detection(temp_audio_path)
            suggestion_voice = suggest_action_voice(result_voice)
            st.write(f"Detected Emotion: {result_voice.upper()}")
            st.write(f"Suggested Action: {suggestion_voice}")
            os.remove(temp_audio_path)

    

st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 10px;
        width: 100%;
        text-align: center;
        color: gray;
        font-size: 14px;
    }
    </style>
    <div class="footer">
        <p>This prototype is developed as part of the application for Postdoctoral Researcher or Senior Research Fellow in Data Science at the University of Oulu.</p>
    </div>
    """, unsafe_allow_html=True)
