import os
#print(os.listdir('../../../secrets'))
import tensorflow as tf
import tensorflow_datasets as tfds
import requests
from google.cloud import storage
from PIL import Image
import numpy as np

# Constants
IMG_SIZE = 128  # Resize images to 128x128 for faster processing
BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', 'default-bucket-name')  # Retrieve from environment variable
#print(os.listdir('os.getcwd()'))
os.environ['GOOGLE_APPLICATION_CREDENTIALS']='~/Users/peter/Desktop/AC215/GitClones/secrets/data-service-account.json'

# URL of the nutritions dataset to download
dataset_url = "https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/nutrients.csv"

# Local file path to save the downloaded nutritions dataset
local_file_path = "nutrients.csv"

def preprocess_and_upload(image, label):
    """Preprocess a single image and upload it to GCS."""
    img = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.uint8)
    img_array = np.array(img)
    pil_img = Image.fromarray(img_array)

#    label_str = label.decode("utf-8")
    label_str = label.encode().decode()
#    blob_name = f"{label_str}/{tf.random.uniform(shape=[], minval=1, maxval=1e7, dtype=tf.int32)}.jpg"

    # Initialize Google Cloud Storage client
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)

    blob = bucket.blob('dfsdadfsfads')
    blob.upload_from_string(pil_img.tobytes(), content_type='image/jpeg')

def preprocess_data():
    """Load and preprocess the Food-101 dataset."""
    (ds_train, ds_test), ds_info = tfds.load(
        'cifar10',
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
    
# Function to download the nutritions dataset from the URL
def download_nutritions(url, local_path):
    response = requests.get(url)
    response.raise_for_status()
    
    with open(local_path, "wb") as f:
        f.write(response.content)

# Function to upload a file to Google Cloud Storage
def upload_to_gcs(bucket_name, source_file, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file)

if __name__ == "__main__":
    preprocess_data()
    
    # Download the nutritions dataset
    download_dataset(dataset_url, local_file_path)
    # Upload the downloaded nutritions dataset to GCS
    upload_to_gcs(gcs_bucket_name, local_file_path, gcs_object_name)
    print(f"Preprocessed data uploaded to GCS bucket {BUCKET_NAME}")
