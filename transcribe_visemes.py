from fastapi import FastAPI, File, UploadFile
import uvicorn
import tempfile
import os
from allosaurus.app import read_recognizer, download_model
from allosaurus.model import get_model_path
from allosaurus.lm.inventory import Inventory
import torch
import shutil
from whisper import load_model
from typing import List, Dict

from app import app

# Download the Allosaurus English model if not already present
model_name = "eng2102"
download_model(model_name)

# Initialize the Allosaurus recognizer with the English-only model
recognizer = read_recognizer(model_name)

# List the phonemes supported by the model using Inventory
inventory = Inventory(get_model_path(model_name))
supported_phonemes = list(inventory.unit.id_to_unit.values())[1:]
print("Supported Phonemes:", ' '.join(supported_phonemes))

# Define the phoneme-to-viseme mapping based on OVRLipSync reference
phoneme_to_viseme = {
    'a': 'aa',
    'b': 'PP',
    'd': 'DD',
    'd͡ʒ': 'CH',
    'e': 'E',
    'f': 'FF',
    'h': 'CH',
    'i': 'ih',
    'j': 'ih',
    'k': 'kk',
    'l': 'nn',
    'm': 'PP',
    'n': 'NN',
    'o': 'oh',
    'p': 'PP',
    's': 'SS',
    't': 'DD',
    't͡ʃ': 'CH',
    'u': 'oh',
    'v': 'FF',
    'w': 'ou',
    'z': 'SS',
    'æ': 'aa',
    'ð': 'TH',
    'ŋ': 'NN',
    'ɑ': 'aa',
    'ɔ': 'oh',
    'ə': 'ou',
    'ɛ': 'E',
    'ɡ': 'kk',
    'ɪ': 'ih',
    'ɹ': 'RR',
    'ɹ̩': 'RR',
    'ʃ': 'CH',
    'ʊ': 'ou',
    'ʌ': 'ou',
    'ʒ': 'CH',
    'θ': 'TH'
}

async def transcribe_visemes(audio_path: str):
    # Perform phoneme recognition using Allosaurus with timestamps
    results = recognizer.recognize(audio_path, timestamp=True).splitlines()

    # Collect phonemes, their timestamps, and corresponding visemes
    phoneme_data = []
    for result in results:
        parts = result.split()
        if len(parts) == 3:
            start_time = float(parts[0])
            duration = float(parts[1])
            phoneme = parts[2]
            end_time = start_time + duration
            viseme = phoneme_to_viseme.get(phoneme, 'Unknown')
            phoneme_data.append({
                "phoneme": phoneme,
                "viseme": viseme,
                "start_time": start_time,
                "end_time": end_time
            })

    return {"transcription": phoneme_data}

@app.post("/transcribe-to-phonemes/")
async def transcribe_to_phonemes(file: UploadFile = File(...)):
    """
    Endpoint to transcribe an uploaded wav file to phonemes with timestamps and visemes.
    """
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        audio_path = tmp.name

    visemes = await transcribe_visemes(audio_path)

    # remove temporary file
    os.remove(audio_path)

    return visemes

# Load the Whisper model once during startup
model = load_model("base")  # You can replace "base" with any other model size you have downloaded