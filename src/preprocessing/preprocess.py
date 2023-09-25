import tensorflow as tf
import tensorflow_datasets as tfds
import os
from google.cloud import storage
from PIL import Image
import numpy as np

# Constants
IMG_SIZE = 128  # Resize images to 128x128 for faster processing
BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', 'default-bucket-name')  # Retrieve from environment variable
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

def preprocess_and_upload(image, label_str):
    """Preprocess a single image and upload it to GCS."""
    # Resize and cast the image
    img = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.uint8)

    # Encode tensor image to JPEG string
    img_encoded = tf.image.encode_jpeg(img)

    blob_name = f"{label_str}/{tf.random.uniform(shape=[], minval=1, maxval=int(1e7), dtype=tf.int32)}.jpg"

    # Initialize Google Cloud Storage client
    client = storage.Client.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)
    bucket = client.get_bucket(BUCKET_NAME)

    blob = bucket.blob(blob_name)
    blob.upload_from_string(img_encoded.numpy(), content_type='image/jpeg')

def preprocess_data():
    """Load and preprocess the Food-101 dataset."""
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # Process training data
    for img, label in ds_train.take(500):  # Adjust the number as needed
        label_str = ds_info.features['label'].int2str(label)
        preprocess_and_upload(img, label_str)

    # Process testing data
    for img, label in ds_test.take(100):  # Adjust the number as needed
        label_str = ds_info.features['label'].int2str(label)
        preprocess_and_upload(img, label_str)

if __name__ == "__main__":
    preprocess_data()
    print(f"Preprocessed data uploaded to GCS bucket {BUCKET_NAME}.")
