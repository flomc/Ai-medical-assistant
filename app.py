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
import json
from gtts import gTTS
import tempfile  # To create temporary files
from faster_whisper import WhisperModel  # Importing WhisperModel from faster-whisper

# Load environment variables
load_dotenv()

# Tune Studio API headers
TUNE_STUDIO_URL = "https://proxy.tune.app/chat/completions"
TUNE_STUDIO_HEADERS = {
    "Authorization": "sk-tune-2tieMUjQ6tZJmCaz3SRL9h6oeOSkxhrH7ys",  # Replace with your actual Tune Studio API key
    "Content-Type": "application/json",
}

st.set_page_config(page_title="Nurture Nest", page_icon=":robot:")

st.title("Nurture Nest")
st.subheader("An application that can help new parents take better care of their baby.")

# Initialize Whisper model from faster-whisper
model = WhisperModel("base")  # You can choose the model size (e.g., "small", "medium", "large")

# Record audio
audio_bytes = audio_recorder()

# Inject custom CSS to center elements
st.markdown(
    """
    <style>
    .centered-audio {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Check if audio recording was successful
if audio_bytes:
    # Center the audio button using the custom CSS class
    st.markdown('<div class="centered-audio">', unsafe_allow_html=True)
    st.audio(audio_bytes, format="audio/wav")
    st.markdown('</div>', unsafe_allow_html=True)

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

        # Transcribe audio using faster-whisper
        st.write("Transcribing audio...")
        segments, info = model.transcribe(wav_buffer, beam_size=5)

        # Combine transcriptions from all segments
        transcription = " ".join([segment.text for segment in segments])
        st.write(f"Transcription: {transcription}")

        # Prepare the data to send to the Tune Studio chatbot
        chatbot_data = {
            "temperature": 0.8,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a newborn and infant care specialist."
                },
                {
                    "role": "user",
                    "content": transcription + " Make it specific. Do not include any markdown. Please make your response brief."  # Use the transcribed text as the prompt for the chatbot
                }
            ],
            "model": "rohan/gemma-2-27b-it",
            "stream": False,
            "frequency_penalty": 0,
            "max_tokens": 900
        }

        # Send the transcription as input to the Tune Studio chatbot
        chatbot_response = requests.post(
            TUNE_STUDIO_URL,
            headers=TUNE_STUDIO_HEADERS,
            json=chatbot_data
        )

        # Check if the chatbot request was successful
        if chatbot_response.status_code == 200:
            chatbot_output = chatbot_response.json()

            # Extract the content from the chatbot response
            hf_tts_inp = chatbot_output.get("choices", [{}])[0].get("message", {}).get("content", "No response received.")
            st.write(f"Chatbot Response: {hf_tts_inp}")

            # Convert the chatbot response to speech using gTTS
            tts = gTTS(text=hf_tts_inp, lang='en', slow=False)

            # Save the TTS output to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
                tts.save(temp_audio_file.name)
                temp_audio_path = temp_audio_file.name

            # Use the operating system to play the audio
            if os.name == 'nt':  # For Windows
                os.system(f'start {temp_audio_path}')
            elif os.name == 'posix':  # For macOS or Linux
                os.system(f'afplay {temp_audio_path}')  # Use afplay on macOS
                # For Linux, you can use mpg123, mplayer, or another command
                # os.system(f'mpg123 {temp_audio_path}')
            else:
                st.warning("Unsupported OS for audio playback through os command.")

        else:
            st.error(f"Failed to connect to the chatbot: {chatbot_response.status_code} - {chatbot_response.text}")
else:
    st.info("Please record your audio using the recorder above.")
