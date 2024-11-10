from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
import os
import torch
import shutil
from whisper import load_model
from typing import List, Dict, Union, Optional

# Create FastAPI instance as a global variable for sharing across multiple files
from app import app

# Load the Whisper model once during startup
model = load_model("base")  # You can replace "base" with any other model size you have downloaded


async def transcribe_file(temp_path: str, transcript: Optional[str] = None) -> List[Dict[str, Union[str, float]]]:
    # Load the audio file and perform transcription or forced alignment
    if transcript:
        # Use forced alignment mode by specifying the transcript
        result = model.transcribe(temp_path, word_timestamps=True, initial_prompt=transcript)
    else:
        # Perform full transcription
        result = model.transcribe(temp_path, word_timestamps=True)

    # Prepare the response data
    words_with_timestamps = []
    for segment in result["segments"]:
        for word in segment["words"]:
            words_with_timestamps.append({
                "word": word["word"],
                "start": str(word["start"]),  # Convert to string to avoid type errors
                "end": str(word["end"])  # Convert to string to avoid type errors
            })

    return words_with_timestamps


@app.post("/transcribe/")
async def transcribe_audio(
        file: UploadFile = File(...),
        transcript: Optional[str] = Form(None)  # Optional transcript input for forced alignment
) -> List[Dict[str, Union[str, float]]]:
    # Save the uploaded file temporarily
    temp_dir = "uploads"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    temp_path = os.path.join(temp_dir, file.filename)
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Call the transcribe_file function
    words_with_timestamps = await transcribe_file(temp_path, transcript)

    # Delete the temporary audio file
    os.remove(temp_path)

    return words_with_timestamps


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
