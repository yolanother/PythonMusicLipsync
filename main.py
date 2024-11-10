from app import app, encode_json_and_file, log
import tempfile
from fastapi import UploadFile, File, Form
from typing import List, Dict, Union, Optional
from timestamped_transcription import transcribe_file
from transcribe_visemes import transcribe_visemes
import wave
import struct
import json
from io import BytesIO
from vocal_remover.api import *
import uvicorn
from beat_detection import *

def encode_json_and_file(json_data, wav_path):
    FLAG_SIZE = 1
    LONG_SIZE = 8

    # Convert JSON data to a byte array
    json_bytes = json.dumps(json_data).encode('utf-8') if json_data else b''
    json_length = len(json_bytes)

    # Read and convert the WAV file content to PCM in-memory
    with wave.open(wav_path, 'rb') as wav_file:
        params = wav_file.getparams()
        channels = params.nchannels
        sample_width = params.sampwidth
        frame_rate = params.framerate
        num_frames = params.nframes
        file_bytes = wav_file.readframes(num_frames)

    binary_length = len(file_bytes)

    # Set flags
    has_binary = 1 if binary_length > 0 else 0
    has_json = 1 if json_length > 0 else 0

    # Create the flag byte
    flags = (has_binary | (has_json << 1))

    # Construct the byte array with header
    header = struct.pack(
        f'B{LONG_SIZE}s{LONG_SIZE}s',
        flags,
        struct.pack('q', json_length),  # 'q' is for 8-byte signed long
        struct.pack('q', binary_length)  # 'q' for 8-byte signed long
    )

    # Concatenate header, JSON bytes, and file bytes
    encoded_data = header + json_bytes + file_bytes

    return encoded_data

# function to convert float timestamp to ms. Multiply by 1000 and round then convert to string
def convert_to_ms(timestamp):
    # if timestamp is a string convert to float
    if isinstance(timestamp, str):
        timestamp = float(timestamp)
    return int(timestamp * 1000)


@app.post("/analyze/")
async def process_audio(
        file: UploadFile = File(...),
        transcript: Optional[str] = Form(None)  # Optional transcript input for forced alignment
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        audio_path = tmp.name

    log("analyze", "Extracting vocals...")
    vocals = get_vocals(audio_path)

    log("analyze", "Transcribing visemes...")
    visemes = await transcribe_visemes(vocals[0])

    log("analyze", "Transcribing text...")
    transcript_times = await transcribe_file(vocals[0], transcript)

    log("analyze", "Analyzing beats...")
    beats = await analyze_beat(audio_path)

    # delete all files in the vocals
    for vocal in vocals:
        os.remove(vocal)

    data = []

    log("analyze", "Combining data...")
    # append beats if we have them
    for beat in beats:
        data.append(beat)

    for i in range(len(transcript_times)):
        # if the transcript time has a caption, add it to the data
        if "caption" in transcript_times[i]:
            data.append({
                "data": transcript_times[i]["caption"],
                "offset": convert_to_ms(transcript_times[i]["start"]),
                "type": "CAPTION"
            })
        else:
            # trim space off the word
            word = transcript_times[i]["word"].strip()
            data.append({
                "data": word,
                "offset": convert_to_ms(transcript_times[i]["start"]),
                "type": "WORD"
            })

    for i in range(len(visemes["transcription"])):
        data.append({
            "data": visemes["transcription"][i]["phoneme"],
            "offset": convert_to_ms(visemes["transcription"][i]["start_time"]),
            "type": "PHONE"
        })
        data.append({
            "data": visemes["transcription"][i]["viseme"],
            "offset": convert_to_ms(visemes["transcription"][i]["start_time"]),
            "type": "VISEME"
        })

    log("analyze", "Sorting data...")
    # sort the data by start time
    data.sort(key=lambda x: x["offset"])

    # print the data with pretty json
    log("analyze", json.dumps(data, indent=2))

    encoded = encode_json_and_file(data, audio_path)

    # remove temporary file
    os.remove(audio_path)

    # Return as a streaming response for binary data
    return StreamingResponse(BytesIO(encoded), media_type="application/octet-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
