from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form
import struct
import json

# Create FastAPI instance as a global variable for sharing across multiple files
app = FastAPI()


def log(endpoint, message):
    # print message [endpoint name - timestamp] message
    print(f"[{endpoint} - {datetime.now()}] {message}")


def encode_json_and_file(json_data, file_path):
    FLAG_SIZE = 1
    LONG_SIZE = 8

    # Convert JSON data to a byte array
    json_bytes = json.dumps(json_data).encode('utf-8') if json_data else b''
    json_length = len(json_bytes)

    # Read the binary file content
    with open(file_path, 'rb') as file:
        file_bytes = file.read()
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
