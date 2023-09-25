import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers.experimental import preprocessing
import os
from google.cloud import storage
from PIL import Image
import numpy as np

# Constants
IMG_SIZE = 128
BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', 'default-bucket-name')
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

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

def preprocess_and_upload(image, label_str, split):
    img = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.uint8)
    img_encoded = tf.image.encode_jpeg(img)

    # Adjust blob_name to have "train", "val", or "test" and then class folder
    blob_name = f"{split}/{label_str}/{tf.random.uniform(shape=[], minval=1, maxval=int(1e7), dtype=tf.int32)}.jpg"

    client = storage.Client.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(img_encoded.numpy(), content_type='image/jpeg')

def preprocess_data():
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        'food101',
        split=['train', 'validation[:10100]', 'validation[10100:]'],
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

    for img_batch, label_batch in ds_train.take(1):
        for img, label in zip(img_batch, label_batch):
            label_str = ds_info.features['label'].int2str(label.numpy())
            preprocess_and_upload(img, label_str, 'train')

    for img_batch, label_batch in ds_val.take(1):
        for img, label in zip(img_batch, label_batch):
            label_str = ds_info.features['label'].int2str(label.numpy())
            preprocess_and_upload(img, label_str, 'val')

    for img_batch, label_batch in ds_test.take(1):
        for img, label in zip(img_batch, label_batch):
            label_str = ds_info.features['label'].int2str(label.numpy())
            preprocess_and_upload(img, label_str, 'test')


if __name__ == "__main__":
    preprocess_data()
    print(f"Preprocessed and augmented data uploaded to GCS bucket {BUCKET_NAME}.")
