import base64
import sys
import traceback

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

def encode_json_and_file(json_data, audio_path, output_format="pcm", sample_rate=24000, channels=1):
    FLAG_SIZE = 1
    LONG_SIZE = 8

    # Convert JSON data to a byte array
    json_bytes = json.dumps(json_data).encode('utf-8') if json_data else b''
    json_length = len(json_bytes)

    # Read and convert the audio file content based on the requested format
    audio = AudioSegment.from_file(audio_path)
    if output_format == "wav":
        audio = audio.set_frame_rate(sample_rate).set_sample_width(2).set_channels(channels)
        file_bytes = audio.export(format="wav").read()
    elif output_format == "mp3":
        audio = audio.set_frame_rate(sample_rate).set_sample_width(2).set_channels(channels)
        file_bytes = audio.export(format="mp3").read()
    else:  # default to pcm
        # reencode the audio to PCM with 16k sample rate, 16-bit sample width, and mono channel
        audio = audio.set_frame_rate(sample_rate).set_sample_width(2).set_channels(channels)
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
        include_base64: Optional[bool] = Form(False),  # Include base64 encoded audio in JSON response
        sample_rate: Optional[int] = Form(24000),  # Specify the sample rate for the audio
        channels: Optional[int] = Form(1)  # Specify the number of channels for the audio
):
    log("analyze", f"Received file {file.filename}")
    try:
        # get a name for the file based on the md5 of the file
        name = hashlib.md5(file.filename.encode()).hexdigest()
        uploads_dir = 'uploads'
        os.makedirs(uploads_dir, exist_ok=True)
        os.makedirs('output_vocals', exist_ok=True)
        audio_path = f'uploads/{name}'
        # write the file to disk
        with open(audio_path, 'wb') as f:
            # Read the uploaded file and convert it to WAV if necessary
            audio = AudioSegment.from_file(BytesIO(await file.read()))
            audio.export(audio_path, format="wav")

        log("analyze", "Extracting vocals...")
        vocals = get_vocals(audio_path)

        log("analyze", "Transcribing visemes...")
        visemes = await transcribe_visemes(vocals[0])

        log("analyze", "Transcribing text...")
        transcript_times = await transcribe_file(vocals[0], transcript)

        log("analyze", "Analyzing beats...")
        beats = await analyze_beat(audio_path)

        # delete all files in the vocals
        #for vocal in vocals:
        #    os.remove(vocal)

        data = []
        words = []
        viseme_list = []

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

                entry = {
                    "data": word,
                    "time": convert_to_ms(transcript_times[i]["start"]),
                    "type": type
                }
                data.append(entry)
                if type == "WORD":
                    words.append(entry)

        for i in range(len(visemes["transcription"])):
            data.append({
                "data": visemes["transcription"][i]["phoneme"],
                "time": convert_to_ms(visemes["transcription"][i]["start_time"]),
                "type": "PHONE"
            })
            start_time = convert_to_ms(visemes["transcription"][i]["start_time"])
            end_time = convert_to_ms(visemes["transcription"][i]["end_time"])
            viseme = {
                "data": visemes["transcription"][i]["viseme"],
                "time": start_time,
                "length": end_time - start_time,
                "type": "VISEME"
            }
            data.append(viseme)
            viseme_list.append(viseme)

        # iterate over data for VISEMEs if there is more than 500ms between visemes add a sil viseme
        for i in range(len(viseme_list) - 1):
            delta = viseme_list[i + 1]["time"] - (viseme_list[i]["time"] + viseme_list[i]["length"])
            if delta > 100:
                data.append({
                    "data": "sil",
                    "time": viseme_list[i]["time"] + viseme_list[i]["length"],
                    "type": "VISEME"
                })
                data.append({
                    "data": "sil",
                    "time": viseme_list[i + 1]["time"] - 10,
                    "type": "VISEME"
                })

        # add a sil after the last viseme
        data.append({
            "data": "sil",
            "time": viseme_list[-1]["time"] + viseme_list[-1]["length"],
            "type": "VISEME"
        })

        if len(words) > 2:
            # iterate over the words, if there is more than 5 sec between them put a instrumental emote right after the last word
            # of the break put a vocal emote right before the next word
            for i in range(len(words) - 1):
                delta = words[i + 1]["time"] - words[i]["time"]
                if delta > 5000:
                    data.append({
                        "data": "instrumental",
                        "time": words[i]["time"],
                        "type": "EMOTE"
                    })
                    data.append({
                        "data": "vocal",
                        "time": max(0, words[i + 1]["time"] - 250),
                        "type": "EMOTE"
                    })
            # if there is more than 5 sec before the first word put a instrumental emote at the beginning, otherwise put the vocal emote
            if words[0]["time"] > 5000:
                data.insert(0, {
                    "data": "instrumental",
                    "time": 0,
                    "type": "EMOTE"
                })
                # insert a vocal at the time of the first word
                data.insert(1, {
                    "data": "vocal",
                    "time": words[0]["time"],
                    "type": "EMOTE"
                })
            else:
                data.insert(0, {
                    "data": "vocal",
                    "time": words[0]["time"],
                    "type": "EMOTE"
                })

            # add a instrumental emote at the last word
            data.append({
                "data": "instrumental",
                "time": words[-1]["time"] + 250,
                "type": "EMOTE"
            })


        # Iterate over the data and calculate the sample offset based on the time and the sample rate of 24000 single channel
        for d in data:
            d["offset"] = int(d["time"] * sample_rate / 1000 * channels)

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
                data = encode_json_and_file(data, audio_path, output_format=output_format, sample_rate=sample_rate, channels=channels)
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
        traceback.print_exc()
        log("analyze", f"Error processing audio: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    #if args --init is passed, don't start the server
    if "--init" not in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8000)
