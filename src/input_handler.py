import pytesseract
from PIL import Image
import os
from dotenv import load_dotenv
from groq import Groq

# Explicitly tell Python where the Tesseract worker lives on your Windows machine
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load environment variables (API keys)
load_dotenv()

# Initialize Groq client instead of OpenAI (used for free Whisper ASR)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def process_image(image_file):
    """
    Accepts JPG/PNG images [cite: 36] and performs OCR to extract text[cite: 36].
    Returns extracted text and a mock confidence score.
    """
    try:
        img = Image.open(image_file)
        # Perform OCR using Tesseract 
        extracted_text = pytesseract.image_to_string(img)
        
        # Note: Tesseract doesn't give a simple global confidence score natively without complex parsing.
        # For the sake of the assignment's HITL trigger, we will simulate a confidence score
        # based on text length/quality to trigger HITL if confidence is low[cite: 38].
        confidence_score = 0.9 if len(extracted_text.strip()) > 5 else 0.4
        
        return extracted_text.strip(), confidence_score
    except Exception as e:
        return f"Error processing image: {str(e)}", 0.0

def process_audio(audio_file_path):
    """
    Accepts audio [cite: 42] and converts speech to text (ASR) [cite: 43] using Whisper via Groq.
    """
    try:
        # Using Groq's Whisper model for ASR 
        with open(audio_file_path, "rb") as audio_file:
            # Groq's API expects the file parameter as a tuple: (filename, file_bytes)
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3", # Groq's supported free Whisper model
                file=(audio_file_path, audio_file.read()),
                prompt="mathematics, equations, square root of, raised to, integral, derivative" # Handles math-specific phrases [cite: 45]
            )
        
        # Whisper is generally highly accurate, but we assign a mock confidence
        # to test the HITL flow if needed.
        confidence_score = 0.95 
        
        return transcription.text, confidence_score
    except Exception as e:
        return f"Error processing audio: {str(e)}", 0.0

def process_text(text_input):
    """
    Normal typed input[cite: 50].
    """
    # Text is inherently 100% confident as it is directly typed by the user.
    return text_input.strip(), 1.0