# Python Music Lipsync
This project provides a Python-based API to create a lipsync file that can be played back using the Voice SDK's speech system.

## Features
- Generate lipsync files from audio input
- Compatible with Voice SDK's speech system

## Requirements
- Python
- pip

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/PythonMusicLipsync.git
    cd PythonMusicLipsync
    ```

2. Install the required dependencies using pip:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. Run the script using Python:
    ```sh
    python -u main.py
    ```

## API Documentation

### Transcribe Audio
**Endpoint:** `/transcribe/`  
**Method:** `POST`  
**Summary:** Transcribe an uploaded audio file.  
**Request Body:**
- `file` (required): The audio file to be transcribed.
- `transcript` (optional): An optional transcript of the audio file.

**Responses:**
- `200 OK`: Successful response containing the transcription.
- `422 Unprocessable Entity`: Validation error.

### Transcribe To Phonemes
**Endpoint:** `/transcribe-to-phonemes/`  
**Method:** `POST`  
**Summary:** Transcribe an uploaded wav file to phonemes with timestamps and visemes.  
**Request Body:**
- `file` (required): The wav file to be transcribed.

**Responses:**
- `200 OK`: Successful response.
- `422 Unprocessable Entity`: Validation error.

### Extract Vocals
**Endpoint:** `/extract-vocals/`  
**Method:** `POST`  
**Summary:** Extract vocals from an uploaded audio file.  
**Request Parameters:**
- `download` (optional): Boolean flag to indicate if the result should be downloadable.

**Request Body:**
- `uploaded_file` (required): The audio file from which to extract vocals.

**Responses:**
- `200 OK`: Successful response.
- `422 Unprocessable Entity`: Validation error.

### Extract Tracks Zip
**Endpoint:** `/extract-tracks-zip/`  
**Method:** `POST`  
**Summary:** Extract tracks from an uploaded zip file.  
**Request Body:**
- `uploaded_file` (required): The zip file containing tracks to be extracted.

**Responses:**
- `200 OK`: Successful response.
- `422 Unprocessable Entity`: Validation error.

### Process Audio
**Endpoint:** `/process/`  
**Method:** `POST`  
**Summary:** Process an uploaded audio file. This file can be a fully composed music audio file. It must be a wav file.  
**Request Body:**
- `file` (required): The audio file to be processed.
- `transcript` (optional): An optional transcript of the audio file.

**Responses:**
- `200 OK`: Successful response.
- `422 Unprocessable Entity`: Validation error.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.