import hashlib
import traceback
import argparse

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import shutil
import os

import librosa
import numpy as np
import soundfile as sf
import torch

from pathlib import Path
from tempfile import NamedTemporaryFile
import zipfile

from vocal_remover.lib import nets
from vocal_remover.lib import spec_utils

from typing import List
from vocal_remover.inference import Separator

from app import app

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models')
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, 'baseline.pth')
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device(f'cuda:0')

p = argparse.ArgumentParser()
p.add_argument('--init', '-i', action='store_true')
p.add_argument('--gpu', '-g', type=int, default=-1)
p.add_argument('--pretrained_model', '-P', type=str, default=DEFAULT_MODEL_PATH)
p.add_argument('--sr', '-r', type=int, default=44100)
p.add_argument('--n_fft', '-f', type=int, default=2048)
p.add_argument('--hop_length', '-H', type=int, default=1024)
p.add_argument('--batchsize', '-B', type=int, default=4)
p.add_argument('--cropsize', '-c', type=int, default=256)
p.add_argument('--output_image', '-I', action='store_true')
p.add_argument('--tta', '-t', action='store_true')
p.add_argument('--complex', '-X', action='store_true')
args = p.parse_args()
model = nets.CascadedNet(args.n_fft, args.hop_length, 32, 128, args.complex)
model.load_state_dict(torch.load(args.pretrained_model, map_location='cpu'))
model.to(device)

# Function to extract vocals from an audio file
def get_vocals(audio_file_path: str, n_fft: int = 2048, hop_length: int = 512, sr: int = 44100, batchsize: int = 4, cropsize: int = 512, tta: bool = False) -> List[str]:
    try:
        basename = os.path.splitext(os.path.basename(audio_file_path))[0]
        output_dir = './output_vocals/'
        Path(output_dir).mkdir(exist_ok=True)

        instruments_path = f'{output_dir}{basename}_Instruments.wav'
        vocals_path = f'{output_dir}{basename}_Vocals.wav'
        print("Getting vocals for file:", audio_file_path)

        # if vocals exists return it
        if os.path.exists(vocals_path):
            return [vocals_path, instruments_path]

        # Loading wave source
        X, sr = librosa.load(audio_file_path, sr=sr, mono=False, dtype=np.float32, res_type='kaiser_fast')

        if X.ndim == 1:
            # mono to stereo
            X = np.asarray([X, X])

        # STFT of wave source
        X_spec = spec_utils.wave_to_spectrogram(X, hop_length, n_fft)

        sp = Separator(
            model=model,
            device=device,
            batchsize=batchsize,
            cropsize=cropsize
        )

        if tta:
            y_spec, v_spec = sp.separate_tta(X_spec)
        else:
            y_spec, v_spec = sp.separate(X_spec)


        # Inverse STFT of instruments and vocals
        wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=hop_length)
        sf.write(instruments_path, wave.T, sr)

        wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=hop_length)

        sf.write(vocals_path, wave.T, sr)

        return [vocals_path, instruments_path]
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Unexpected error: {e}")


# Endpoint to handle uploading an audio file, extracting vocals, and streaming/download
@app.post("/extract-vocals/")
async def extract_vocals(uploaded_file: UploadFile = File(...), download: bool = False):
    try:
        # get a name for the file based on the md5 of the file
        name = hashlib.md5(uploaded_file.filename.encode()).hexdigest()
        path = f'./uploads/{name}'
        # write the file to disk
        with open(f'./uploads/{name}', 'wb') as f:
            f.write(await uploaded_file.read())

        vocals_files = get_vocals(path)
        if not vocals_files:
            raise HTTPException(status_code=404, detail="No vocals extracted.")

        # Stream the first vocal file if it exists
        def iterfile():
            with open(vocals_files[0], mode="rb") as file_like:
                yield from file_like

        if download:
            return StreamingResponse(iterfile(), headers={
                "Content-Disposition": f"attachment; filename={uploaded_file.filename}_vocals.wav"
            })
        else:
            return StreamingResponse(iterfile(), media_type="audio/wav")
    except Exception as e:
        # print the full exception
        print(e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        #if tmp_path and os.path.exists(tmp_path):
        #    os.unlink(tmp_path)
        # Clean up any generated output
        output_folder = Path("./output_vocals/") / Path(tmp_path).stem
        #if output_folder.exists():
        #    shutil.rmtree(output_folder)


# Endpoint to handle uploading an audio file, extracting all tracks, and creating a zip file
@app.post("/extract-tracks-zip/")
async def extract_tracks_zip(uploaded_file: UploadFile = File(...)):
    try:
        with NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await uploaded_file.read())
            tmp_path = tmp.name

        # Extract tracks (vocals and accompaniment) from the provided audio file
        tracks_files = get_vocals(tmp_path, pretrained_model='models/baseline.pth', gpu=-1, n_fft=2048, hop_length=512, sr=44100, complex=True, batchsize=4, cropsize=512, tta=False)
        if not tracks_files:
            raise HTTPException(status_code=404, detail="No tracks extracted.")

        # Create a zip file containing all extracted tracks
        zip_filename = f"{Path(tmp_path).stem}_tracks.zip"
        with NamedTemporaryFile(delete=False, suffix=".zip") as zip_tmp:
            with zipfile.ZipFile(zip_tmp, 'w') as zipf:
                for track_file in tracks_files:
                    zipf.write(track_file, Path(track_file).name)
            zip_path = zip_tmp.name

        # Stream the zip file as a response
        def iterfile():
            with open(zip_path, mode="rb") as file_like:
                yield from file_like

        return StreamingResponse(iterfile(), headers={
            "Content-Disposition": f"attachment; filename={zip_filename}"
        }, media_type="application/zip")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if zip_path and os.path.exists(zip_path):
            os.unlink(zip_path)
        # Clean up any generated output
        output_folder = Path("./output_vocals/") / Path(tmp_path).stem
        if output_folder.exists():
            shutil.rmtree(output_folder)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
