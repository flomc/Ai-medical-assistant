# importing neccseary modules
import streamlit as st
from pathlib import Path
import os
from audio_recorder_streamlit import audio_recorder
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

import torch
import soundfile as sf

# Load the pre-trained model and processor


from dotenv import load_dotenv
load_dotenv()

#from apikey import api_key1
# set the page configuration
api_key1 = os.getenv('api_key1')
st.set_page_config(page_title="VitalImage Analytics", page_icon=":robot:")

# config gemini api key

# Set up the model
generation_config = {
  "temperature": 1,
  "top_p": 1,
  "top_k": 32,
  "max_output_tokens": 8192,
}

# saftey setting  

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]


system_prompt="""

 As a highly skilled medical practitioner specializing in image analysis, you are tasked with examining medical images for renowed hospital .Your expertise is crucial in  identifying any anomolies ,diseases ,or health issues 



 Your Responsibilities include :
1.Detailed Analysis: Thoroughly analyze each image, focusing on identifying any abnormal find

2.Findings Report: Document all observed anomalies or signs of disease. Clearly articulate these findings

 3.Recommendations and Next Steps: Based on your analysis, suggest potential next steps, including future test or treatments  that are required


 4.Treatment Suggestions: If appropriate, recommend possible treatment options or intervention



 1.Scope of Response: Only respond if the image pertains to human health issues.
2. Clarity of Image: In cases where the image quality impedes clear analysis, note that certain ascepts are "unable to determine based on provided iamge"
3. Disclaimer: Accompany your analysis with the disclaimer: "Consult with a Doctor before making any decision.


4.Your insights are invaluable in guiding clinical decisions. Please proceed with the analysis  ,adhering to strucre approach ouline above


Please provide me an output with these 4 heading 1)detailed analysis ,2)Findings Report 3)Recommendations and Next Steps 4)Treatment Suggestions


"""

# model configaration 


#set the logo

#set the logo
st.image(r"ai-assistant--that--looks-like--nurse.png", width=150)

#set the title

st.title(" Ai 🤖 Medical Assistant 👨‍⚕⚕️👩‍⚕")

#set the subtitle

st.subheader("An application that can help users to identify medical images")

audio_bytes = audio_recorder()
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Load your audio file
audio_input, sample_rate = sf.read(audio_bytes)

# Preprocess the audio input
input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values

# Perform inference (transcribe the audio)
with torch.no_grad():
    logits = model(input_values).logits

# Get the predicted IDs
predicted_ids = torch.argmax(logits, dim=-1)

# Decode the IDs to text
transcription = processor.decode(predicted_ids[0])

st.write(transcription)
    

   
