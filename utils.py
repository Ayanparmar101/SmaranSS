from openai import OpenAI
import os
from dotenv import load_dotenv
import base64
import streamlit as st
import wave  # For checking audio file duration

# Load environment variables
load_dotenv()

# Get the OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")  # Ensure this matches the .env file
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please check your .env file.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

def get_answer(messages):
    system_message = [{
        "role": "system", 
        "content": """
        You are a Socratic tutor. Never give direct answers. 
        Ask thought-provoking questions to guide users to discover answers independently. 
        Use analogies, open-ended prompts, and step-by-step inquiry. 
        Example responses:
        - User: 'What is gravity?' → Bot: 'What happens when you drop an object? How does Earth pull things toward it?'
        - User: 'Why is the sky blue?' → Bot: 'What happens when sunlight enters Earth’s atmosphere? How do colors behave in a prism?'
        """
    }]
    messages = system_message + messages
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        max_tokens=75  
    )
    return response.choices[0].message.content

def get_audio_duration(file_path):
    """Get the duration of an audio file in seconds."""
    try:
        with wave.open(file_path, "rb") as audio_file:
            frames = audio_file.getnframes()
            rate = audio_file.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return 0

def speech_to_text(audio_data):
    # Check if the file exists
    if not os.path.exists(audio_data):
        raise FileNotFoundError(f"Audio file not found: {audio_data}")

    # Check file size (Whisper API limit is 25 MB)
    file_size = os.path.getsize(audio_data)  # Size in bytes
    if file_size > 25 * 1024 * 1024:  # 25 MB in bytes
        raise ValueError("File size exceeds 25 MB limit.")

    # Check audio duration (must be at least 0.1 seconds)
    duration = get_audio_duration(audio_data)
    if duration < 0.1:
        raise ValueError("Audio file is too short. Minimum audio length is 0.1 seconds.")

    try:
        with open(audio_data, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                response_format="text",
                file=audio_file
            )
        return transcript
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def text_to_speech(input_text):
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=input_text
    )
    webm_file_path = "temp_audio_play.mp3"
    with open(webm_file_path, "wb") as f:
        response.stream_to_file(webm_file_path)
    return webm_file_path

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)