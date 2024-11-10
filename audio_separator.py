from fastapi import FastAPI, UploadFile, HTTPException, Response
from fastapi.responses import StreamingResponse
import torch
import torchaudio
from speechbrain.pretrained import SepformerSeparation as separator
import os

from app import app

# Load the pretrained model
model = separator.from_hparams(
    source="speechbrain/sepformer-wham",
    savedir="pretrained_models/sepformer-wham"
)


# Method to perform audio separation
def separate_audio(file_path: str):
    try:
        # Load the audio file with specifying backend
        mixture_waveform, sample_rate = torchaudio.load(file_path, format='wav')

        # Perform separation
        estimated_sources = model.separate_batch(mixture_waveform)
        return estimated_sources, sample_rate
    except Exception as e:
        raise RuntimeError(f"Audio separation failed: {str(e)}")


@app.post("/separate-audio")
async def separate_audio_endpoint(file: UploadFile):
    output_files = []  # Ensure output_files is always initialized
    try:
        # Ensure uploaded file is an audio file
        if not file.content_type.startswith("audio"):
            raise HTTPException(status_code=400, detail="File is not a valid audio type.")

        # Save file to disk
        file_location = f"temp/{file.filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Perform separation using the reusable method
        separated_sources, sample_rate = separate_audio(file_location)

        # Prepare to return separated files
        for idx, source in enumerate(separated_sources):
            output_file_path = f"temp/separated_source_{idx}.wav"
            torchaudio.save(output_file_path, source, sample_rate)
            output_files.append(output_file_path)

        # Create a generator to stream the files as multipart
        def iter_files(files):
            for file_path in files:
                with open(file_path, "rb") as file:
                    yield from file
                    yield b"\n--boundary\n"

        headers = {
            "Content-Type": "multipart/form-data; boundary=boundary"
        }
        return StreamingResponse(iter_files(output_files), headers=headers)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Cleanup the saved file after processing
        if os.path.exists(file_location):
            os.remove(file_location)
        for output_file in output_files:
            if os.path.exists(output_file):
                os.remove(output_file)


# Example usage of reusable method outside of FastAPI
if __name__ == "__main__":
    # Example file path
    example_path = "example_audio.wav"
    if os.path.exists(example_path):
        try:
            separated, rate = separate_audio(example_path)
            print("Audio separation completed successfully.")
        except RuntimeError as e:
            print(str(e))
