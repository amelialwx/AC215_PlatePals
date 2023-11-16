import os
import json
import numpy as np
import tensorflow as tf
from google.cloud import aiplatform
import base64

AUTOTUNE = tf.data.experimental.AUTOTUNE
labels_path = "/persistent/labels.json"
image_width = 128
image_height = 128
num_channels = 3

def load_preprocess_image_from_path(image_path):
    print("Image", image_path)

    # Prepare the data
    def load_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=num_channels)
        image = tf.image.resize(image, [image_height, image_width])
        return image

    # Normalize pixels
    def normalize(image):
        image = image / 255
        return image

    test_data = tf.data.Dataset.from_tensor_slices(([image_path]))
    test_data = test_data.map(load_image, num_parallel_calls=AUTOTUNE)
    test_data = test_data.map(normalize, num_parallel_calls=AUTOTUNE)
    test_data = test_data.repeat(1).batch(1)

    return test_data

def make_prediction_vertexai(image_path, index2label):
    print("Predict using Vertex AI endpoint")

    # Get the endpoint
    endpoint = aiplatform.Endpoint(
        "projects/939067285486/locations/us-central1/endpoints/3828572055683465216"
    )

    with open(image_path, "rb") as f:
        data = f.read()
    b64str = base64.b64encode(data).decode("utf-8")
    instances = [{"bytes_inputs": {"b64": b64str}}]

    result = endpoint.predict(instances=instances)

    print("Result:", result)
    prediction = result.predictions[0]
    print(prediction, prediction.index(max(prediction)))

    # with open(labels_path, "r") as file:
    #     index2label = json.load(file)

    prediction_label = index2label[str(prediction.index(max(prediction)))]

    return {
        "prediction_label": prediction_label,
        "prediction": prediction,
        "accuracy": round(np.max(prediction) * 100, 2),
    }
