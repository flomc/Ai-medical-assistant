# Import necessary modules
import streamlit as st
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
from dotenv import load_dotenv
import requests
import io
import json
from gtts import gTTS
import tempfile
import torch.nn.functional as F
from faster_whisper import WhisperModel
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch
import sounddevice as sd
import soundfile as sf

# Load environment variables
load_dotenv()

# Tune Studio API headers
TUNE_STUDIO_URL = "https://proxy.tune.app/chat/completions"
TUNE_STUDIO_HEADERS = {
    "Authorization": "sk-tune-2tieMUjQ6tZJmCaz3SRL9h6oeOSkxhrH7ys",
    "Content-Type": "application/json",
}

# Initialize Whisper model
model = WhisperModel("base")

# Load model and processor for baby cry classification
cry_processor = Wav2Vec2Processor.from_pretrained("./fine-tuned-model")
cry_model = Wav2Vec2ForSequenceClassification.from_pretrained("./fine-tuned-model")

# Streamlit page configuration
st.set_page_config(page_title="Nurture Nest", page_icon=":robot:")

st.title("Nurture Nest")
st.subheader("An application that can help new parents take better care of their baby.")

# Set up tabs
tab1, tab2 = st.tabs(["Get Parenting Advice", "Why is my Baby Crying?"])

def classify_audio_from_buffer(audio_buffer):
    sample_rate, audio_input = wavfile.read(audio_buffer)
    if len(audio_input.shape) > 1:
        audio_input = np.mean(audio_input, axis=1).astype(np.float32)

    inputs = cry_processor(audio_input, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    inputs = inputs.to(torch.float32)

    with torch.no_grad():
        logits = cry_model(**inputs).logits

    probabilities = F.softmax(logits, dim=-1)
    predicted_id = torch.argmax(probabilities, dim=-1).item()
    label_names = [
        "Your baby likely has belly pain!", 
        "Your baby needs burping!", 
        "Your baby is uncomfortable because it needs changing!", 
        "Your baby is hungry!", 
        "Your baby is tired!"
    ]
    predicted_label = label_names[predicted_id]

    return predicted_label


# Function to record audio for a fixed duration (7 seconds)
def record_audio(duration=7, sample_rate=16000):
    """Records audio for a specified duration and saves it to a temporary file."""
    #st.write(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    temp_wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    sf.write(temp_wav_path, audio, sample_rate)
    return temp_wav_path

# Function to play audio using os commands based on the operating system
def play_audio(audio_path):
    """Plays audio using os system commands based on the OS."""
    try:
        if os.name == 'nt':  # For Windows
            os.system(f'start {audio_path}')
        elif os.name == 'posix':
            if 'darwin' in os.uname().sysname.lower():  # For macOS
                os.system(f'afplay {audio_path}')
            else:  # For Linux
                os.system(f'mpg123 {audio_path}')  # Assuming mpg123 is installed
    except Exception as e:
        st.warning(f"Failed to play audio: {e}")

# Tab 1: Get Parenting Advice
with tab1:
    # Button to record audio for advice
    if st.button("Record Audio for Advice"):
        audio_path = record_audio(duration=7)  # Record for 7 seconds
        st.audio(audio_path, format="audio/wav")

        # Process the audio file as required (transcription, sending to chatbot, etc.)
        sample_rate, audio_input = wavfile.read(audio_path)

        # Convert to mono if stereo
        if len(audio_input.shape) > 1:
            audio_input = np.mean(audio_input, axis=1).astype(np.int16)

        # Check the current sample rate and resample if necessary
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            num_samples = int(len(audio_input) * float(target_sample_rate) / sample_rate)
            audio_input = resample(audio_input, num_samples).astype(np.int16)
            sample_rate = target_sample_rate

        # Ensure the audio input is sufficiently long
        if len(audio_input) < 16000:
            st.error("Audio is too short for processing. Please record a longer audio clip.")
        else:
            wav_buffer = io.BytesIO()
            wavfile.write(wav_buffer, sample_rate, audio_input)
            wav_buffer.seek(0)

            with st.spinner(" "):
                segments, info = model.transcribe(wav_buffer, beam_size=5)
                transcription = " ".join([segment.text for segment in segments])
                #st.write(f"Transcription: {transcription}")

            chatbot_data = {
                "temperature": 0.8,
                "messages": [
                    {"role": "system", "content": "You are a newborn and infant care specialist. You will keep your responses specific and brief and not include any markdown."},
                    {"role": "user", "content": transcription + "Please make your response brief. Do not include any markdown."}
                ],
                "model": "janetm/tdata2-model-8h3t0fxz",
                "stream": False,
                "frequency_penalty": 0,
                "max_tokens": 900
            }

            with st.spinner("Contacting the parenting advice AI..."):
                chatbot_response = requests.post(TUNE_STUDIO_URL, headers=TUNE_STUDIO_HEADERS, json=chatbot_data)

            if chatbot_response.status_code == 200:
                chatbot_output = chatbot_response.json()
                hf_tts_inp = chatbot_output.get("choices", [{}])[0].get("message", {}).get("content", "No response received.")
                st.write(f"Response: {hf_tts_inp}")

                tts = gTTS(text=hf_tts_inp, lang='en', slow=False)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
                    tts.save(temp_audio_file.name)
                    temp_audio_path = temp_audio_file.name

                play_audio(temp_audio_path)
            else:
                st.error(f"Failed to connect to the chatbot: {chatbot_response.status_code} - {chatbot_response.text}")

# Tab 2: Why is my Baby Crying?
with tab2:
    st.write("Record your baby's cry to find out what is causing them to do so.")

    # Button to record baby's cry
    if st.button("Record Baby's Cry"):
        audio_path = record_audio(duration=7)  # Record for 7 seconds
        st.audio(audio_path, format="audio/wav")

        # Process and classify the baby's cry
        sample_rate, audio_input = wavfile.read(audio_path)

        # Convert to mono if stereo
        if len(audio_input.shape) > 1:
            audio_input = np.mean(audio_input, axis=1).astype(np.int16)

        # Check the current sample rate and resample if necessary
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            num_samples = int(len(audio_input) * float(target_sample_rate) / sample_rate)
            audio_input = resample(audio_input, num_samples).astype(np.int16)
            sample_rate = target_sample_rate

        # Ensure the audio input is sufficiently long
        if len(audio_input) < 16000:
            st.error("Audio is too short for processing. Please record a longer audio clip.")
        else:
            wav_buffer = io.BytesIO()
            wavfile.write(wav_buffer, sample_rate, audio_input)
            wav_buffer.seek(0)

            with st.spinner("Analyzing your baby's cry..."):
                category = classify_audio_from_buffer(wav_buffer)
                st.markdown(
                    f"""
                    <div style="text-align: center; margin-top: 20px;">
                        <h1 style="color: #ff4b4b; font-size: 32px;">{category}</h1>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

# Function to classify audio data from a BytesIO object
