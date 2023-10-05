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
    "--wandb_key", dest="wandb_key", default="16", type=str, help="WandB API key."
)
parser.add_argument(
    "--bucket_name", dest="bucket_name", default="default_bucket_name", type=str, help="GCS bucket name."
)
args = parser.parse_args()

# TF Version
print("tensorflow version", tf.__version__)
print("Eager Execution Enabled:", tf.executing_eagerly())

# Get the number of replicas
strategy = tf.distribute.MirroredStrategy()
print("Number of replicas:", strategy.num_replicas_in_sync)

# See number of devices
devices = tf.config.experimental.get_visible_devices()
print("Devices:", devices)
print(tf.config.experimental.list_logical_devices("GPU"))

print("GPU Available: ", tf.config.list_physical_devices("GPU"))
print("All Physical Devices", tf.config.list_physical_devices())


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
bucket_name = args.bucket_name
data_version = "preprocessed_data" # CHANGE THIS
splits = ['train', 'val', 'test'] 

# Ensure that destination directories exist or create them
for split in splits:
    destination_path = os.path.join('./data')
    os.makedirs(destination_path, exist_ok=True)
    
    # Construct the blob name
    blob_name = f"{data_version}/{split}.zip"
    
    # Download and unzip
    download_and_unzip_from_gcs(bucket_name, blob_name, destination_path)

end_time = time.time()
duration = (end_time - start_time) / 60
print(f"Download execution time {duration} minutes.")


# Function to create image paths and labels
def gather_data_from_directory(data_dir):
    image_paths = []
    labels = []

    for class_label in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_label)
        
        # Check if the path is a directory, skip if not
        if not os.path.isdir(class_path):
            continue

        for image_name in os.listdir(class_path):
            image_paths.append(os.path.join(class_path, image_name))
            labels.append(class_label)

    return image_paths, labels


# Function to turn labels into indexes
def encode_labels(labels):
    unique_labels = sorted(set(labels))
    label2index = {label: index for index, label in enumerate(unique_labels)}
    encoded_labels = [label2index[label] for label in labels]

    return encoded_labels, label2index

# Obtain data
train_x, train_y = gather_data_from_directory("./data/train")
val_x, val_y = gather_data_from_directory("./data/val")
test_x, test_y = gather_data_from_directory("./data/test")

# Get all labels and indexed labels
all_labels = train_y + val_y + test_y
all_encoded_labels, label2index = encode_labels(all_labels)

# Filter for train and val labels and indexed labels
train_y_encoded = all_encoded_labels[:len(train_y)]
val_y_encoded = all_encoded_labels[len(train_y):len(train_y) + len(val_y)]
test_y_encoded = all_encoded_labels[len(train_y) + len(val_y):]
num_classes = len(label2index)

# Sanity check
print("train_x count:", len(train_x))
print("validate_x count:", len(val_x))
print("test_x count:", len(test_x))
print("total classes:", num_classes)

# Login into wandb
wandb.login(key=args.wandb_key)


# Create TF Data
def get_dataset(image_width=128, image_height=128, num_channels=3, batch_size=32, num_classes = 101):
    # Load Image
    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=num_channels)
        image = tf.image.resize(image, [image_height, image_width])
        return image, label

    # Normalize pixels
    def normalize(image, label):
        image = image / 255
        return image, label

    train_shuffle_buffer_size = len(train_x)
    validation_shuffle_buffer_size = len(val_x)

    # Converts to y to binary class matrix (One-hot-encoded)
    #train_processed_y = to_categorical(train_y_encoded, num_classes=num_classes, dtype="float32")
    #validate_processed_y = to_categorical(val_y_encoded, num_classes=num_classes, dtype="float32")
    #test_processed_y = to_categorical(test_y_encoded, num_classes=num_classes, dtype="float32")

    # Create TF Dataset
    # train_data = tf.data.Dataset.from_tensor_slices((train_x, train_processed_y))
    # validation_data = tf.data.Dataset.from_tensor_slices((val_x, validate_processed_y))
    # test_data = tf.data.Dataset.from_tensor_slices((test_x, test_processed_y))

    # Create TF Dataset
    train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y_encoded))
    validation_data = tf.data.Dataset.from_tensor_slices((val_x, val_y_encoded))
    test_data = tf.data.Dataset.from_tensor_slices((test_x, test_y_encoded))

    # Train data
    train_data = train_data.shuffle(buffer_size=train_shuffle_buffer_size)
    train_data = train_data.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    #train_data = train_data.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    train_data = train_data.batch(batch_size)
    train_data = train_data.prefetch(tf.data.AUTOTUNE)

    # Validation data
    validation_data = validation_data.shuffle(buffer_size=validation_shuffle_buffer_size)
    validation_data = validation_data.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    #validation_data = validation_data.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    validation_data = validation_data.batch(batch_size)
    validation_data = validation_data.prefetch(tf.data.AUTOTUNE)

    # Test data
    test_data = test_data.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    #test_data = test_data.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    test_data = test_data.batch(batch_size)
    test_data = test_data.prefetch(tf.data.AUTOTUNE)

    return (train_data, validation_data, test_data)

 
# MobileNet model
def build_mobilenet_model(
    image_height, image_width, num_channels, num_classes, model_name, train_base=False
):
    # Model input
    input_shape = [image_height, image_width, num_channels]  # height, width, channels

    # Load a pretrained model from keras.applications
    tranfer_model_base = keras.applications.MobileNetV2(
        input_shape=input_shape, weights="imagenet", include_top=False
    )

    # Freeze the mobileNet model layers
    tranfer_model_base.trainable = train_base

    # Regularize using L1
    # kernel_weight = 0.02
    # bias_weight = 0.02
    kernel_weight = 0.0
    bias_weight = 0.0

    model = Sequential(
        [
            tranfer_model_base,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(
                units=128,
                activation="relu",
                kernel_regularizer=keras.regularizers.l1(kernel_weight),
                bias_regularizer=keras.regularizers.l1(bias_weight),
            ),
            keras.layers.Dense(
                units=num_classes,
                activation="softmax",
                kernel_regularizer=keras.regularizers.l1(kernel_weight),
                bias_regularizer=keras.regularizers.l1(bias_weight),
            ),
        ],
        name=model_name + "_train_base_" + str(train_base),
    )

    return model


# Efficient net model
def build_efficient_net(image_height, image_width, num_channels, num_classes, model_name, train_base = False):

    input_shape = (image_height,image_width,num_channels)
    base_model = tf.keras.applications.EfficientNetB0(include_top = False)
    base_model.trainable= train_base

    # Create functional model
    inputs = keras.layers.Input(shape=input_shape, name= "input_layer")

    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(num_classes)(x)
    outputs = keras.layers.Activation("softmax", dtype=tf.float32, name ="softmax_float32")(x)
    model = tf.keras.Model(inputs,outputs, name = model_name + "_train_base_" + str(train_base))

    #Get a summary of model
    return model


# In class model
def build_model_tfhub(
    image_height, image_width, num_channels, num_classes, model_name, train_base=False
):
    # Model input
    input_shape = [image_height, image_width, num_channels]  # height, width, channels

    # Handle to pretrained model
    handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"

    # Regularize using L1
    kernel_weight = 0.02
    bias_weight = 0.02

    model = Sequential(
        [
            keras.layers.InputLayer(input_shape=input_shape),
            hub.KerasLayer(handle, trainable=train_base),
            keras.layers.Dense(
                units=64,
                activation="relu",
                kernel_regularizer=keras.regularizers.l1(kernel_weight),
                bias_regularizer=keras.regularizers.l1(bias_weight),
            ),
            keras.layers.Dense(
                units=num_classes,
                activation="softmax",
                kernel_regularizer=keras.regularizers.l1(kernel_weight),
                bias_regularizer=keras.regularizers.l1(bias_weight),
            ),
        ],
        name=model_name + "_train_base_" + str(train_base),
    )

    return model


print("Train model")
############################
# Training Params
############################
model_name = args.model_name
learning_rate = 0.001
image_width = 128
image_height = 128
num_channels = 3
batch_size = args.batch_size
epochs = args.epochs
train_base = args.train_base

# Free up memory
K.clear_session()

# Data
train_data, validation_data, test_data = get_dataset(
    image_width=image_width,
    image_height=image_height,
    num_channels=num_channels,
    batch_size=batch_size,
    num_classes=num_classes 
)
print("Converted to Tensorflow dataset.")

if model_name == "mobilenetv2":
    # Model
    model = build_mobilenet_model(
        image_height,
        image_width,
        num_channels,
        num_classes,
        model_name,
        train_base=train_base,
    )
    # Optimizer
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    # Loss
    loss = keras.losses.categorical_crossentropy
    # Print the model architecture
    print(model.summary())
    # Compile
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

elif model_name == "tfhub_mobilenetv2":
    # Model
    model = build_model_tfhub(
        image_height,
        image_width,
        num_channels,
        num_classes,
        model_name,
        train_base=train_base,
    )
    # Optimizer
    #optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer = keras.optimizers.Adam()
    # Loss
    loss = keras.losses.sparse_categorical_crossentropy
    # Print the model architecture
    print(model.summary())
    # Compile
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

elif model_name == "EfficientNetV2B0":
    # Model
    model = build_efficient_net(
        image_height,
        image_width,
        num_channels,
        num_classes,
        model_name,
        train_base=train_base,
    )
    # Optimizer
    #optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer = keras.optimizers.Adam()
    # Loss
    loss = keras.losses.sparse_categorical_crossentropy
    # Print the model architecture
    print(model.summary())
    # Compile
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

#Initialize a W&B run
wandb.init(
    project="platepals-training-vertex-ai",
    config={
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "model_name": model.name,
    },
    name=model.name,
)

# Train model
start_time = time.time()
training_results = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=epochs,
    callbacks=[WandbCallback()],
    verbose=1,
)
execution_time = (time.time() - start_time) / 60.0
print("Training execution time (mins)", execution_time)

# Update W&B
wandb.config.update({"execution_time": execution_time})
# Close the W&B run
wandb.run.finish()

print("Training Job Complete")
