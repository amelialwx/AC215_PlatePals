import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers.experimental import preprocessing
import os
from google.cloud import storage
from PIL import Image
import numpy as np
import zipfile
import requests
import io

# Constants
IMG_SIZE = 128
BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', 'default-bucket-name')
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
dataset_url = "https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/nutrients.csv"
gcs_object_name = "nutrients.csv"
local_file_path = "nutrients.csv"

# Data augmentation layer
Data_augmentation = tf.keras.Sequential([
    preprocessing.RandomFlip('horizontal'),
    preprocessing.RandomRotation(0.2),
    preprocessing.RandomZoom(0.2),
    preprocessing.RandomHeight(0.2),
    preprocessing.RandomWidth(0.2)
], name="data_augmentation")

def preprocess_img(image, label, image_shape=IMG_SIZE):
    image = Data_augmentation(image)
    image = tf.image.resize(image, [image_shape, image_shape])
    return tf.cast(image, tf.float32), label

def create_zip_and_upload(ds, ds_info, split):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for img_batch, label_batch in ds.take(1):
            for img, label in zip(img_batch, label_batch):
                label_str = ds_info.features['label'].int2str(label.numpy())
                file_name = f"{split}/{label_str}/{tf.random.uniform(shape=[], minval=1, maxval=int(1e7), dtype=tf.int32)}.jpg"
                
                # Cast the image to uint8 before encoding
                img_uint8 = tf.cast(img, tf.uint8)
                img_encoded = tf.image.encode_jpeg(img_uint8)
                
                zf.writestr(file_name, img_encoded.numpy())

    zip_buffer.seek(0)
    blob_name = f"{split}.zip"

    client = storage.Client.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.upload_from_file(zip_buffer, content_type='application/zip')

def preprocess_data():
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        'food101',
        split=['train', 'validation[:40%]', 'validation[40%:]'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)

    ds_train = ds_train.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(tf.data.AUTOTUNE)
    ds_val = ds_val.batch(batch_size=32).prefetch(buffer_size=1000)
    ds_test = ds_test.batch(batch_size=32).prefetch(buffer_size=1000)

    create_zip_and_upload(ds_train, ds_info, 'train')
    create_zip_and_upload(ds_val, ds_info, 'val')
    create_zip_and_upload(ds_test, ds_info, 'test')

def download_nutrients_dataset(url, local_path):
    response = requests.get(url)
    response.raise_for_status()
    
    with open(local_path, "wb") as f:
        f.write(response.content)

def upload_nutrients_to_gcs(bucket_name, source_file, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file)

if __name__ == "__main__":
    preprocess_data()
    download_nutrients_dataset(dataset_url, local_file_path)
    upload_nutrients_to_gcs(BUCKET_NAME, local_file_path, gcs_object_name)
    print(f"Preprocessed and augmented data uploaded to GCS bucket {BUCKET_NAME}.")
