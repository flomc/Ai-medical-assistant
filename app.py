# Importing necessary modules
import streamlit as st
from pathlib import Path
import os
from audio_recorder_streamlit import audio_recorder
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from scipy.io import wavfile
from scipy.signal import resample
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load API key from environment variables
api_key1 = os.getenv('api_key1')
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

    # Save the audio bytes to a temporary file
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_bytes)

    # Load the pre-trained model and processor
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    # Load your audio file using scipy
    sample_rate, audio_input = wavfile.read("temp_audio.wav")

    # Convert to mono if stereo
    if len(audio_input.shape) > 1:
        audio_input = np.mean(audio_input, axis=1)

    # Check the current sample rate and resample if necessary
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        # Resample the audio to 16000 Hz
        num_samples = int(len(audio_input) * float(target_sample_rate) / sample_rate)
        audio_input = resample(audio_input, num_samples)
        sample_rate = target_sample_rate

    # Ensure the audio input is sufficiently long
    if len(audio_input) < 16000:  # 1 second of audio at 16kHz
        st.error("Audio is too short for processing. Please record a longer audio clip.")
    else:
        # Preprocess the audio input
        input_values = processor(
            np.array(audio_input, dtype=np.float32),
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).input_values

        # Perform inference (transcribe the audio)
        with torch.no_grad():
            logits = model(input_values).logits

        # Get the predicted IDs
        predicted_ids = torch.argmax(logits, dim=-1)

        # Decode the IDs to text
        transcription = processor.decode(predicted_ids[0])

        # Display the transcription
        st.write(transcription)
else:
    st.info("Please record your audio using the recorder above.")
