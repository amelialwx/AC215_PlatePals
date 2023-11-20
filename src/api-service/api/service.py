from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import asyncio
import pandas as pd
import os
from fastapi import File
from tempfile import TemporaryDirectory
from api import model
import json
import requests
import zipfile
import tensorflow as tf

# Initialize Tracker Service
# tracker_service = TrackerService()

# Setup FastAPI app
app = FastAPI(title="API Server", description="API Server", version="v1")

# Enable CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
# @app.get("/")
# async def get_index():
#     return {"message": "Welcome to the API Service"}

@app.get("/")
async def get_index():
    labels_path = "/persistent/labels.json"
    try:
        with open(labels_path, "r") as file:
            labels = json.load(file)
        return {"message": "Welcome to the API Service", "labels": labels}
    except FileNotFoundError:
        return {"message": "Welcome to the API Service", "error": "labels.json not found"}


@app.post("/predict")
async def predict(file: bytes = File(...)):
    labels_path = "/persistent/labels.json"
    print("predict file:", len(file), type(file))

    def download_file(packet_url, base_path="", extract=False, headers=None):
        if base_path != "":
            if not os.path.exists(base_path):
                os.mkdir(base_path)
        packet_file = os.path.basename(packet_url)
        with requests.get(packet_url, stream=True, headers=headers) as r:
            r.raise_for_status()
            with open(os.path.join(base_path, packet_file), "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        if extract:
            if packet_file.endswith(".zip"):
                with zipfile.ZipFile(os.path.join(base_path, packet_file)) as zfile:
                    zfile.extractall(base_path)
            else:
                packet_name = packet_file.split(".")[0]
                with tarfile.open(os.path.join(base_path, packet_file)) as tfile:
                    tfile.extractall(base_path)

    self_host_model = True
    
    if self_host_model:
        download_file(
            "https://github.com/amelialwx/models/releases/download/v2.0/model.zip",
            base_path="artifacts",
            extract=True,
        )
        artifact_dir = "./artifacts/model"

        # Load model
        prediction_model = tf.keras.models.load_model(artifact_dir) 

    # Save the image
    with TemporaryDirectory() as image_dir:
        image_path = os.path.join(image_dir, "test.png")
        with open(image_path, "wb") as output:
            output.write(file)

        # Make prediction
        prediction_results = {}
        with open(labels_path, "r") as file:
            labels = json.load(file)
        if self_host_model:
            prediction_results = model.make_prediction(image_path, prediction_model, labels)
        else:
            with open(labels_path, "r") as file:
                labels = json.load(file)
            prediction_results = model.make_prediction_vertexai(image_path, labels)

    print(prediction_results)
    return prediction_results
