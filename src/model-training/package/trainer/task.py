import argparse
import os
import requests
import zipfile
import tarfile
import time
from google.cloud import storage

# Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.layer_utils import count_params

# sklearn
from sklearn.model_selection import train_test_split

# Tensorflow Hub
import tensorflow_hub as hub

# W&B
import wandb
from wandb.keras import WandbCallback, WandbMetricsLogger


# Setup the arguments for the trainer task
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-dir", dest="model_dir", default="test", type=str, help="Model dir."
)
parser.add_argument("--lr", dest="lr", default=0.001, type=float, help="Learning rate.")
parser.add_argument(
    "--model_name",
    dest="model_name",
    default="mobilenetv2",
    type=str,
    help="Model name",
)
parser.add_argument(
    "--train_base",
    dest="train_base",
    default=False,
    action="store_true",
    help="Train base or not",
)
parser.add_argument(
    "--epochs", dest="epochs", default=10, type=int, help="Number of epochs."
)
parser.add_argument(
    "--batch_size", dest="batch_size", default=16, type=int, help="Size of a batch."
)
parser.add_argument(
    "--wandb_key", dest="wandb_key", default="16", type=str, help="WandB API Key"
)
args = parser.parse_args()

# TF Version
print("tensorflow version", tf.__version__)
print("Eager Execution Enabled:", tf.executing_eagerly())
# Get the number of replicas
strategy = tf.distribute.MirroredStrategy()
print("Number of replicas:", strategy.num_replicas_in_sync)

devices = tf.config.experimental.get_visible_devices()
print("Devices:", devices)
print(tf.config.experimental.list_logical_devices("GPU"))

print("GPU Available: ", tf.config.list_physical_devices("GPU"))
print("All Physical Devices", tf.config.list_physical_devices())


# # Utils functions
# def download_file(packet_url, base_path="", extract=False, headers=None):
#     if base_path != "":
#         if not os.path.exists(base_path):
#             os.mkdir(base_path)
#     packet_file = os.path.basename(packet_url)
#     with requests.get(packet_url, stream=True, headers=headers) as r:
#         r.raise_for_status()
#         with open(os.path.join(base_path, packet_file), "wb") as f:
#             for chunk in r.iter_content(chunk_size=8192):
#                 f.write(chunk)

#     if extract:
#         if packet_file.endswith(".zip"):
#             with zipfile.ZipFile(os.path.join(base_path, packet_file)) as zfile:
#                 zfile.extractall(base_path)
#         else:
#             packet_name = packet_file.split(".")[0]
#             with tarfile.open(os.path.join(base_path, packet_file)) as tfile:
#                 tfile.extractall(base_path)


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

def download_and_unzip_from_gcs(bucket_name, blob_name, destination_path):
    """
    Download and unzip a blob from Google Cloud Storage.

    Parameters:
        bucket_name (str): The name of the bucket.
        blob_name (str): The name of the blob (object) to download.
        destination_path (str): The path where the downloaded blob should be saved and extracted.

    Returns:
        None
    """
    # Ensure that GOOGLE_APPLICATION_CREDENTIALS is set in environment variables
    #assert 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ, "GOOGLE_APPLICATION_CREDENTIALS is not set in the environment variables."
    
    # Initialize a storage client and get the bucket and blob
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Prepare the path to save the downloaded zip file
    zip_path = os.path.join(destination_path, blob_name.split('/')[-1])
    
    # Download the blob to the zip_path
    blob.download_to_filename(zip_path)
    print(f"Blob {blob_name} downloaded to {zip_path}.")
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination_path)
    
    # Remove the downloaded zip file after extraction
    os.remove(zip_path)
    print(f"Data from {zip_path} extracted to {destination_path}.")


# Download Data
start_time = time.time()
# download_file(
#     "https://github.com/dlops-io/datasets/releases/download/v1.0/mushrooms_3_labels.zip",
#     base_path="datasets",
#     extract=True,
# )
bucket_name = os.environ.get('GCS_BUCKET_DATA_URI', 'default-bucket-name')
data_version = "preprocessed_data"  # Example version
splits = ['train', 'val', 'test']  # Example splits

# Ensure that destination directories exist or create them
for split in splits:
    destination_path = os.path.join('./data', split)
    os.makedirs(destination_path, exist_ok=True)
    
    # Construct the blob name according to your previous structure
    blob_name = f"{data_version}/{split}.zip"
    
    # Download and unzip
    download_and_unzip_from_gcs(bucket_name, blob_name, destination_path)
end_time = time.time()
duration = (end_time - start_time) / 60
print(f"Download execution time {duration} minutes.")
# # Load Data
# base_path = os.path.join("datasets", "mushrooms")
# label_names = os.listdir(base_path)
# print("Labels:", label_names)

# # Number of unique labels
# num_classes = len(label_names)
# # Create label index for easy lookup
# label2index = dict((name, index) for index, name in enumerate(label_names))
# index2label = dict((index, name) for index, name in enumerate(label_names))

# # Generate a list of labels and path to images
# data_list = []
# for label in label_names:
#     # Images
#     image_files = os.listdir(os.path.join(base_path, label))
#     data_list.extend([(label, os.path.join(base_path, label, f)) for f in image_files])

# print("Full size of the dataset:", len(data_list))
# print("data_list:", data_list[:5])

# # Load X & Y
# # Build data x, y
# data_x = [itm[1] for itm in data_list]
# data_y = [itm[0] for itm in data_list]
# print("data_x:", len(data_x))
# print("data_y:", len(data_y))
# print("data_x:", data_x[:5])
# print("data_y:", data_y[:5])

# # Split Data
# test_percent = 0.10
# validation_percent = 0.2

# # Split data into train / test
# train_validate_x, test_x, train_validate_y, test_y = train_test_split(
#     data_x, data_y, test_size=test_percent
# )

# # Split data into train / validate
# train_x, validate_x, train_y, validate_y = train_test_split(
#     train_validate_x, train_validate_y, test_size=test_percent
# )

# print("train_x count:", len(train_x))
# print("validate_x count:", len(validate_x))
# print("test_x count:", len(test_x))

# # Login into wandb
# wandb.login(key=args.wandb_key)


# # Create TF Datasets
# def get_dataset(image_width=224, image_height=224, num_channels=3, batch_size=32):
#     # Load Image
#     def load_image(path, label):
#         image = tf.io.read_file(path)
#         image = tf.image.decode_jpeg(image, channels=num_channels)
#         image = tf.image.resize(image, [image_height, image_width])
#         return image, label

#     # Normalize pixels
#     def normalize(image, label):
#         image = image / 255
#         return image, label

#     train_shuffle_buffer_size = len(train_x)
#     validation_shuffle_buffer_size = len(validate_x)

#     # Convert all y labels to numbers
#     train_processed_y = [label2index[label] for label in train_y]
#     validate_processed_y = [label2index[label] for label in validate_y]
#     test_processed_y = [label2index[label] for label in test_y]

#     # Converts to y to binary class matrix (One-hot-encoded)
#     train_processed_y = to_categorical(
#         train_processed_y, num_classes=num_classes, dtype="float32"
#     )
#     validate_processed_y = to_categorical(
#         validate_processed_y, num_classes=num_classes, dtype="float32"
#     )
#     test_processed_y = to_categorical(
#         test_processed_y, num_classes=num_classes, dtype="float32"
#     )

#     # Create TF Dataset
#     train_data = tf.data.Dataset.from_tensor_slices((train_x, train_processed_y))
#     validation_data = tf.data.Dataset.from_tensor_slices(
#         (validate_x, validate_processed_y)
#     )
#     test_data = tf.data.Dataset.from_tensor_slices((test_x, test_processed_y))

#     #############
#     # Train data
#     #############
#     # Apply all data processing logic
#     train_data = train_data.shuffle(buffer_size=train_shuffle_buffer_size)
#     train_data = train_data.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
#     train_data = train_data.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
#     train_data = train_data.batch(batch_size)
#     train_data = train_data.prefetch(tf.data.AUTOTUNE)

#     ##################
#     # Validation data
#     ##################
#     # Apply all data processing logic
#     validation_data = validation_data.shuffle(
#         buffer_size=validation_shuffle_buffer_size
#     )
#     validation_data = validation_data.map(
#         load_image, num_parallel_calls=tf.data.AUTOTUNE
#     )
#     validation_data = validation_data.map(
#         normalize, num_parallel_calls=tf.data.AUTOTUNE
#     )
#     validation_data = validation_data.batch(batch_size)
#     validation_data = validation_data.prefetch(tf.data.AUTOTUNE)

#     ############
#     # Test data
#     ############
#     # Apply all data processing logic
#     test_data = test_data.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
#     test_data = test_data.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
#     test_data = test_data.batch(batch_size)
#     test_data = test_data.prefetch(tf.data.AUTOTUNE)

#     return (train_data, validation_data, test_data)


# def build_mobilenet_model(
#     image_height, image_width, num_channels, num_classes, model_name, train_base=False
# ):
#     # Model input
#     input_shape = [image_height, image_width, num_channels]  # height, width, channels

#     # Load a pretrained model from keras.applications
#     tranfer_model_base = keras.applications.MobileNetV2(
#         input_shape=input_shape, weights="imagenet", include_top=False
#     )

#     # Freeze the mobileNet model layers
#     tranfer_model_base.trainable = train_base

#     # Regularize using L1
#     kernel_weight = 0.02
#     bias_weight = 0.02

#     model = Sequential(
#         [
#             tranfer_model_base,
#             keras.layers.GlobalAveragePooling2D(),
#             keras.layers.Dense(
#                 units=128,
#                 activation="relu",
#                 kernel_regularizer=keras.regularizers.l1(kernel_weight),
#                 bias_regularizer=keras.regularizers.l1(bias_weight),
#             ),
#             keras.layers.Dense(
#                 units=num_classes,
#                 activation="softmax",
#                 kernel_regularizer=keras.regularizers.l1(kernel_weight),
#                 bias_regularizer=keras.regularizers.l1(bias_weight),
#             ),
#         ],
#         name=model_name + "_train_base_" + str(train_base),
#     )

#     return model


# def build_model_tfhub(
#     image_height, image_width, num_channels, num_classes, model_name, train_base=False
# ):
#     # Model input
#     input_shape = [image_height, image_width, num_channels]  # height, width, channels

#     # Handle to pretrained model
#     handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"

#     # Regularize using L1
#     kernel_weight = 0.02
#     bias_weight = 0.02

#     model = Sequential(
#         [
#             keras.layers.InputLayer(input_shape=input_shape),
#             hub.KerasLayer(handle, trainable=train_base),
#             keras.layers.Dense(
#                 units=64,
#                 activation="relu",
#                 kernel_regularizer=keras.regularizers.l1(kernel_weight),
#                 bias_regularizer=keras.regularizers.l1(bias_weight),
#             ),
#             keras.layers.Dense(
#                 units=num_classes,
#                 activation="softmax",
#                 kernel_regularizer=keras.regularizers.l1(kernel_weight),
#                 bias_regularizer=keras.regularizers.l1(bias_weight),
#             ),
#         ],
#         name=model_name + "_train_base_" + str(train_base),
#     )

#     return model


# print("Train model")
# ############################
# # Training Params
# ############################
# model_name = args.model_name
# learning_rate = 0.001
# image_width = 224
# image_height = 224
# num_channels = 3
# batch_size = args.batch_size
# epochs = args.epochs
# train_base = args.train_base

# # Free up memory
# K.clear_session()

# # Data
# train_data, validation_data, test_data = get_dataset(
#     image_width=image_width,
#     image_height=image_height,
#     num_channels=num_channels,
#     batch_size=batch_size,
# )

# if model_name == "mobilenetv2":
#     # Model
#     model = build_mobilenet_model(
#         image_height,
#         image_width,
#         num_channels,
#         num_classes,
#         model_name,
#         train_base=train_base,
#     )
#     # Optimizer
#     optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
#     # Loss
#     loss = keras.losses.categorical_crossentropy
#     # Print the model architecture
#     print(model.summary())
#     # Compile
#     model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
# elif model_name == "tfhub_mobilenetv2":
#     # Model
#     model = build_model_tfhub(
#         image_height,
#         image_width,
#         num_channels,
#         num_classes,
#         model_name,
#         train_base=train_base,
#     )
#     # Optimizer
#     optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
#     # Loss
#     loss = keras.losses.categorical_crossentropy
#     # Print the model architecture
#     print(model.summary())
#     # Compile
#     model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

# # Initialize a W&B run
# wandb.init(
#     project="mushroom-training-vertex-ai",
#     config={
#         "learning_rate": learning_rate,
#         "epochs": epochs,
#         "batch_size": batch_size,
#         "model_name": model.name,
#     },
#     name=model.name,
# )

# # Train model
# start_time = time.time()
# training_results = model.fit(
#     train_data,
#     validation_data=validation_data,
#     epochs=epochs,
#     callbacks=[WandbCallback()],
#     verbose=1,
# )
# execution_time = (time.time() - start_time) / 60.0
# print("Training execution time (mins)", execution_time)

# # Update W&B
# wandb.config.update({"execution_time": execution_time})
# # Close the W&B run
# wandb.run.finish()


# print("Training Job Complete")
