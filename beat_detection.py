# Import necessary modules
import os
import tempfile
import traceback

from fastapi import FastAPI, UploadFile, HTTPException
import uvicorn
import numpy as np
import librosa
from BeatNet.BeatNet import BeatNet

# Initialize FastAPI app
from app import app

# Initialize the BeatNet model
estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)

async def analyze_beat(file_path: str):
    """Process the audio file at the given path and return beat JSON data."""
    # Load the audio file using librosa
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None)
    except Exception as e:
        raise Exception(f"Error loading audio: {str(e)}")

    output = estimator.process(file_path)
    # convert the output to the right structure
    beat_json_array = []
    for key,value in output:
        beat_json_array.append({
            "offset": int(float(key) * 1000),
            "data": value,
            "type": "BEAT"
        })

    return beat_json_array

@app.post("/detect-beats/")
async def detect_beats(file: UploadFile):
    try:
        # Ensure the uploaded file is audio
        if not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")

        # Save the uploaded file to a temporary path
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(await file.read())
                audio_path = tmp.name
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error saving audio file: {str(e)}")

        # Process the audio file and return results
        try:
            beat_json_array = await analyze_beat(audio_path)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

        return beat_json_array
    finally:
        # Clean up the temporary audio file
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)

# Run the application if executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
