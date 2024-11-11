import base64
import sys

from app import app, encode_json_and_file, log
import tempfile
from fastapi import UploadFile, File, Form, Request
from typing import List, Dict, Union, Optional
from timestamped_transcription import transcribe_file
from transcribe_visemes import transcribe_visemes
import wave
import struct
import json
from io import BytesIO
from pydub import AudioSegment
from vocal_remover.api import *
import uvicorn
from beat_detection import *
from fastapi.responses import StreamingResponse, JSONResponse

def encode_json_and_file(json_data, audio_path, output_format="pcm"):
    FLAG_SIZE = 1
    LONG_SIZE = 8

    # Convert JSON data to a byte array
    json_bytes = json.dumps(json_data).encode('utf-8') if json_data else b''
    json_length = len(json_bytes)

    # Read and convert the audio file content based on the requested format
    audio = AudioSegment.from_file(audio_path)
    if output_format == "wav":
        audio = audio.set_frame_rate(24000).set_sample_width(2).set_channels(1)
        file_bytes = audio.export(format="wav").read()
    elif output_format == "mp3":
        audio = audio.set_frame_rate(24000).set_sample_width(2).set_channels(1)
        file_bytes = audio.export(format="mp3").read()
    else:  # default to pcm
        # reencode the audio to PCM with 16k sample rate, 16-bit sample width, and mono channel
        audio = audio.set_frame_rate(24000).set_sample_width(2).set_channels(1)
        file_bytes = audio.raw_data

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
        request: Request,
        file: UploadFile = File(...),
        transcript: Optional[str] = Form(None),  # Optional transcript input for forced alignment
        output_format: Optional[str] = Form("pcm"),  # Specify output format: pcm, wav, mp3
        include_base64: Optional[bool] = Form(False)  # Include base64 encoded audio in JSON response
):
    log("analyze", f"Received file {file.filename}")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            # Read the uploaded file and convert it to WAV if necessary
            audio = AudioSegment.from_file(BytesIO(await file.read()))
            audio.export(tmp.name, format="wav")
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
                    "time": convert_to_ms(transcript_times[i]["start"]),
                    "type": "CAPTION"
                })
            else:
                # trim space off the word
                word = transcript_times[i]["word"].strip()
                # if the word is in {emote:trigger-name} it is an action or emote trigger change the type to EMOTE
                type = "WORD"
                if word.startswith("[emote:") and word.endswith("]"):
                    type = "EMOTE"
                    # Replace the word with the values inside the curly braces
                    word = word[7:-1]
                elif word.startswith("[emotion:") and word.endswith("]"):
                    type = "EMOTE"
                    word = word[9:-1]
                elif word.startswith("[action:") and word.endswith("]"):
                    type = "ACTION"
                    # Replace the word with the values inside the curly braces
                    word = word[8:-1]

                data.append({
                    "data": word,
                    "time": convert_to_ms(transcript_times[i]["start"]),
                    "type": type
                })

        for i in range(len(visemes["transcription"])):
            data.append({
                "data": visemes["transcription"][i]["phoneme"],
                "time": convert_to_ms(visemes["transcription"][i]["start_time"]),
                "type": "PHONE"
            })
            data.append({
                "data": visemes["transcription"][i]["viseme"],
                "time": convert_to_ms(visemes["transcription"][i]["start_time"]),
                "type": "VISEME"
            })

        # Iterate over the data and calculate the sample offset based on the time and the sample rate of 24000 single channel
        for d in data:
            d["offset"] = int(d["time"] * 24000 / 1000)

        log("analyze", "Sorting data...")
        # sort the data by start time
        data.sort(key=lambda x: x["offset"])

        # print the data with pretty json
        log("analyze", json.dumps(data, indent=2))

        # Check request type
        if request.headers.get("accept") == "application/json":
            # Return JSON response with data
            response_data = {"data": data}
            if include_base64:
                data = encode_json_and_file(data, audio_path, output_format=output_format)
                base64data = base64.b64encode(data).decode("utf-8")
                response_data["data_encoded_audio"] = base64data
            os.remove(audio_path)
            return JSONResponse(content=response_data)
        else:
            # Return as a streaming response for binary data
            encoded = encode_json_and_file(data, audio_path, output_format=output_format)
            os.remove(audio_path)
            media_type = "audio/wav" if output_format == "wav" else "audio/mpeg" if output_format == "mp3" else "application/octet-stream"
            return StreamingResponse(BytesIO(encoded), media_type=media_type)
    except Exception as e:
        log("analyze", f"Error processing audio: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    #if args --init is passed, don't start the server
    if "--init" not in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8000)
