import tensorflow as tf
import tensorflow_datasets as tfds
import os
from google.cloud import storage
from PIL import Image
import numpy as np

# Constants
IMG_SIZE = 128  # Resize images to 128x128 for faster processing
BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', 'default-bucket-name')  # Retrieve from environment variable

def preprocess_and_upload(image, label):
    """Preprocess a single image and upload it to GCS."""
    img = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.uint8)
    img_array = np.array(img)
    pil_img = Image.fromarray(img_array)

    label_str = label.decode("utf-8")
    blob_name = f"{label_str}/{tf.random.uniform(shape=[], minval=1, maxval=1e7, dtype=tf.int32)}.jpg"

    # Initialize Google Cloud Storage client
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)

    blob = bucket.blob(blob_name)
    blob.upload_from_string(pil_img.tobytes(), content_type='image/jpeg')

def preprocess_data():
    """Load and preprocess the Food-101 dataset."""
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train[:1%]', 'test[:1%]'],
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
    #preprocess_data()
    print(f"Preprocessed data uploaded to GCS bucket {BUCKET_NAME}")
