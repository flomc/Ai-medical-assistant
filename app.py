# Importing necessary modules
import streamlit as st
from pathlib import Path
import os
from audio_recorder_streamlit import audio_recorder
from scipy.io import wavfile
from scipy.signal import resample
import numpy as np
from dotenv import load_dotenv
import requests
import io

# Load environment variables
load_dotenv()

# Set your Hugging Face API token
API_URL = "https://api-inference.huggingface.co/models/distil-whisper/distil-large-v3"
headers = {"Authorization": "Bearer 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"}  # Replace with your actual token

st.set_page_config(page_title="VitalImage Analytics", page_icon=":robot:")

# Set the logo and title
st.image(r"ai-assistant--that--looks-like--nurse.png", width=150)
st.title("AI 🤖 Medical Assistant 👨‍⚕️👩‍⚕️")
st.subheader("An application that can help users to identify medical images")

# Record audio
audio_bytes = audio_recorder()

# Check if audio recording was successful
if audio_bytes:
    # Play the recorded audio
    st.audio(audio_bytes, format="audio/wav")

    # Read the audio bytes using scipy to ensure correct format
    sample_rate, audio_input = wavfile.read(io.BytesIO(audio_bytes))

    # Convert to mono if stereo
    if len(audio_input.shape) > 1:
        audio_input = np.mean(audio_input, axis=1).astype(np.int16)

    # Check the current sample rate and resample if necessary
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        # Resample the audio to 16000 Hz
        num_samples = int(len(audio_input) * float(target_sample_rate) / sample_rate)
        audio_input = resample(audio_input, num_samples).astype(np.int16)
        sample_rate = target_sample_rate

    # Ensure the audio input is sufficiently long
    if len(audio_input) < 16000:  # 1 second of audio at 16kHz
        st.error("Audio is too short for processing. Please record a longer audio clip.")
    else:
        # Save the resampled audio in memory as WAV format
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, sample_rate, audio_input)
        wav_buffer.seek(0)

        # Send the audio data to Hugging Face's Inference API
        response = requests.post(
            API_URL,
            headers=headers,
            data=wav_buffer.read()
        )

        # Check if the request was successful
        if response.status_code == 200:
            transcription = response.json().get("text", "No transcription available.")
            st.write(transcription)
        else:
            st.error(f"Failed to process audio: {response.status_code} - {response.text}")
else:
    st.info("Please record your audio using the recorder above.")
