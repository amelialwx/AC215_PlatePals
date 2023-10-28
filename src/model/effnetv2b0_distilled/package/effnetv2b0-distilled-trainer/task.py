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

# Evaluation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import decimal
from glob import glob
import json


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


def download_and_unzip_from_gcs(bucket_name: str, 
                                blob_name: str, 
                                destination_path: str) -> None:
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


def gather_data_from_directory(data_dir: str) -> Tuple[List[str], List[str]]:
    """
    Create a list of image paths and corresponding labels by scanning a directory.

    Parameters:
        data_dir (str): The directory path to scan for image data.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing lists of image paths and labels.
    """
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


def encode_labels(labels: List[str]) -> Tuple[List[int], Dict[str, int]]:
    """
    Encode a list of labels into integer values and create a label-to-index mapping.

    Parameters:
        labels (List[str]): A list of labels.

    Returns:
        Tuple[List[int], Dict[str, int]]: A tuple containing the encoded labels and the label-to-index mapping.
    """
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


def get_dataset(image_width: int = 128,
                image_height: int = 128,
                num_channels: int = 3,
                batch_size: int = 32,
                num_classes: int = 101) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Create TensorFlow datasets for training, validation, and testing.

    Parameters:
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        num_channels (int): Number of color channels in the image.
        batch_size (int): Batch size for the datasets.
        num_classes (int): Number of target classes.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: A tuple of three TensorFlow datasets for training, validation, and testing.
    """
    
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

    # Create TF Dataset
    train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y_encoded))
    validation_data = tf.data.Dataset.from_tensor_slices((val_x, val_y_encoded))
    test_data = tf.data.Dataset.from_tensor_slices((test_x, test_y_encoded))

    # Train data
    train_data = train_data.shuffle(buffer_size=train_shuffle_buffer_size)
    train_data = train_data.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    # train_data = train_data.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    train_data = train_data.batch(batch_size, drop_remainder=True)
    train_data = train_data.prefetch(tf.data.AUTOTUNE)

    # Validation data
    validation_data = validation_data.shuffle(buffer_size=validation_shuffle_buffer_size)
    validation_data = validation_data.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    # validation_data = validation_data.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    validation_data = validation_data.batch(batch_size, drop_remainder=True)
    validation_data = validation_data.prefetch(tf.data.AUTOTUNE)

    # Test data
    test_data = test_data.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    # test_data = test_data.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    test_data = test_data.batch(batch_size, drop_remainder=True)
    test_data = test_data.prefetch(tf.data.AUTOTUNE)

    print(f'len(train_x): {len(train_x)}')
    print(f'len(test_data): {len(train_data)}')

    return (train_data, validation_data, test_data)


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, decimal.Decimal):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)


experiment_name = "models"
def save_model(model: keras.models.Model,
               model_train_history: dict,
               execution_time: float,
               learning_rate: float,
               epochs: int,
               optimizer: keras.optimizers.Optimizer,
               evaluation_results: List[float]) -> None:
    """
    Save a trained model, its metrics, and training history.

    Parameters:
        model (keras.models.Model): The trained Keras model to be saved.
        model_train_history (dict): Training history of the model.
        execution_time (float): Execution time in minutes.
        learning_rate (float): Learning rate used during training.
        epochs (int): Number of training epochs.
        optimizer (keras.optimizers.Optimizer): Optimizer used for training.
        evaluation_results (List[float]): Model evaluation results.

    Returns:
        None
    """
    model_name=model.name

    # Ensure path exists
    if not os.path.exists(experiment_name):
        os.mkdir(experiment_name)
    # Save the enitire model (structure + weights)
    model.save(os.path.join(experiment_name, model_name+".hdf5"))

    # Save only the weights
    model.save_weights(os.path.join(experiment_name, model_name+".h5"))

    # Save the structure only
    model_json = model.to_json()
    with open(os.path.join(experiment_name, model_name+".json"), "w") as json_file:
        json_file.write(model_json)

    model_size = get_model_size(model_name=model.name)

    # Save model history
    with open(os.path.join(experiment_name, model.name+"_train_history.json"), "w") as json_file:
        json_file.write(json.dumps(model_train_history, cls=JsonEncoder))

    trainable_parameters = count_params(model.trainable_weights)
    non_trainable_parameters = count_params(model.non_trainable_weights)

    # Save model metrics
    metrics = {
        "trainable_parameters":trainable_parameters,
        "execution_time":execution_time,
        "loss":evaluation_results[0],
        "accuracy":evaluation_results[1],
        "model_size":model_size,
        "learning_rate":learning_rate,
        "batch_size":batch_size,
        "epochs":epochs,
        "optimizer":type(optimizer).__name__
    }
    with open(os.path.join(experiment_name,model.name+"_model_metrics.json"), "w") as json_file:
        json_file.write(json.dumps(metrics,cls=JsonEncoder))

        
def get_model_size(model_name: str = "model01") -> int:
    """
    Get the size (in bytes) of a saved model file.

    Parameters:
        model_name (str): Name of the model (default: "model01").

    Returns:
        int: Model size in bytes.
    """
    model_size = os.stat(os.path.join(experiment_name, model_name+".hdf5")).st_size
    return model_size


def append_training_history(
    model_train_history: dict,
    prev_model_train_history: dict,
    metrics: List[str] = ["loss", "val_loss", "accuracy", "val_accuracy"]
) -> dict:
    """
    Append training history metrics from a previous model to the current model's history.

    Parameters:
        model_train_history (dict): Current model's training history.
        prev_model_train_history (dict): Previous model's training history.
        metrics (List[str]): List of metrics to append (default: ["loss", "val_loss", "accuracy", "val_accuracy"]).

    Returns:
        dict: Updated training history with appended metrics.
    """
    for metric in metrics:
        for metric_value in prev_model_train_history[metric]:
            model_train_history[metric].append(metric_value)

    return model_train_history


def evaluate_model(
    model: keras.models.Model,
    test_data: tf.data.Dataset,
    model_train_history: dict,
    execution_time: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    optimizer: keras.optimizers.Optimizer,
    save: bool = True,
    loss_metrics: List[str] = ["loss", "val_loss"],
    acc_metrics: List[str] = ["accuracy", "val_accuracy"]
) -> List[float]:
    """
    Evaluate a trained model on test data, visualize training history, and save the model and metrics.

    Parameters:
        model (keras.models.Model): The trained Keras model to be evaluated.
        test_data (tf.data.Dataset): Test data for evaluation.
        model_train_history (dict): Training history of the model.
        execution_time (float): Execution time in minutes.
        learning_rate (float): Learning rate used during training.
        batch_size (int): Batch size used during training.
        epochs (int): Number of training epochs.
        optimizer (keras.optimizers.Optimizer): Optimizer used for training.
        save (bool): Whether to save the model and metrics (default: True).
        loss_metrics (List[str]): List of loss metrics to visualize and save (default: ["loss", "val_loss"]).
        acc_metrics (List[str]): List of accuracy metrics to visualize and save (default: ["accuracy", "val_accuracy"]).

    Returns:
        List[float]: Model evaluation results, typically [test_loss, test_accuracy].
    """

    # Get the number of epochs the training was run for
    num_epochs = len(model_train_history[loss_metrics[0]])
    
    # Plot training results
    fig = plt.figure(figsize=(15,5))
    axs = fig.add_subplot(1,2,1)
    axs.set_title('Loss')
    # Plot all metrics
    for metric in loss_metrics:
        axs.plot(np.arange(0, num_epochs), model_train_history[metric], label=metric)
    axs.legend()

    axs = fig.add_subplot(1,2,2)
    axs.set_title('Accuracy')
    # Plot all metrics
    for metric in acc_metrics:
        axs.plot(np.arange(0, num_epochs), model_train_history[metric], label=metric)
    axs.legend()

    plt.show()

    # Evaluate on test data
    evaluation_results = model.evaluate(test_data, return_dict=True)

    evaluation_results = [evaluation_results[loss_metrics[0]][-1], evaluation_results[acc_metrics[0]]]

    print("evaluation results:", evaluation_results)
    if save:
        # Save model
        save_model(model, model_train_history,execution_time, learning_rate, epochs, optimizer, evaluation_results)

    return evaluation_results


# MobileNet model
def build_mobilenet_model(image_height: int,
                          image_width: int,
                          num_channels: int,
                          num_classes: int,
                          model_name: str,
                          train_base: bool = False) -> keras.models.Model:
    """
    Build a MobileNet-based Keras model for image classification.

    Parameters:
        image_height (int): Height of the input images.
        image_width (int): Width of the input images.
        num_channels (int): Number of color channels in the input images (e.g., 3 for RGB).
        num_classes (int): Number of target classes for classification.
        model_name (str): Name to assign to the created model.
        train_base (bool, optional): Whether to train the base MobileNet layers (default: False).

    Returns:
        keras.models.Model: A Keras model for image classification based on MobileNet architecture.
    """
    # Model input
    input_shape = [image_height, image_width, num_channels]  # height, width, channels

    # Load a pretrained model from keras.applications
    tranfer_model_base = keras.applications.MobileNetV2(
        input_shape=input_shape, weights="imagenet", include_top=False
    )

    # Freeze the mobileNet model layers
    tranfer_model_base.trainable = train_base

    # Regularize using L1
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
def build_efficient_net(
    image_height: int,
    image_width: int,
    num_channels: int,
    num_classes: int,
    model_name: str,
    train_base: bool = False
) -> keras.models.Model:
    """
    Build a Keras model for image classification using the EfficientNet architecture.

    Parameters:
        image_height (int): Height of the input images.
        image_width (int): Width of the input images.
        num_channels (int): Number of color channels in the input images (e.g., 3 for RGB).
        num_classes (int): Number of target classes for classification.
        model_name (str): Name to assign to the created model.
        train_base (bool, optional): Whether to train the base layers of the EfficientNet model (default: False).

    Returns:
        keras.models.Model: A Keras model for image classification based on the EfficientNet architecture.
    """
    input_shape = (image_height, image_width, num_channels)
    base_model = tf.keras.applications.EfficientNetB0(include_top=False)
    base_model.trainable = train_base

    # Create functional model
    inputs = keras.layers.Input(shape=input_shape, name="input_layer")

    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(num_classes)(x)
    outputs = keras.layers.Activation("softmax", dtype=tf.float32, name="softmax_float32")(x)
    model = tf.keras.Model(inputs, outputs, name=model_name + "_train_base_" + str(train_base))

    #Get a summary of model
    return model


# In class model
def build_model_tfhub(image_height: int,
                      image_width: int,
                      num_channels: int,
                      num_classes: int,
                      model_name: str,
                      train_base: bool = False) -> keras.models.Model
    """
    Build a Keras model for image classification using a TensorFlow Hub (tfhub) pre-trained model.

    Parameters:
        image_height (int): Height of the input images.
        image_width (int): Width of the input images.
        num_channels (int): Number of color channels in the input images (e.g., 3 for RGB).
        num_classes (int): Number of target classes for classification.
        model_name (str): Name to assign to the created model.
        train_base (bool, optional): Whether to train the base layers of the pre-trained model (default: False).

    Returns:
        keras.models.Model: A Keras model for image classification based on a pre-trained model from TensorFlow Hub.
    """
    
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


# Define the Student Model
def build_student_model(image_height, image_width, num_channels, num_classes, model_name):
    input_shape = (image_height, image_width, num_channels)

    # Define the architecture of the student model
    inputs = keras.layers.Input(shape=input_shape, name="input_layer")
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

    # Create the student model
    student_model = keras.models.Model(inputs, outputs, name=model_name)
    return student_model

# Distiller class
class Distiller(Model):
    def __init__(self, teacher, student, model_name="Distiller"):
        super(Distiller, self).__init__(name=model_name)
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, Lambda = 0.1, temperature=3):
      """
      optimizer: Keras optimizer for the student weights
      metrics: Keras metrics for evaluation
      student_loss_fn: Loss function of difference between student predictions and ground-truth
      distillation_loss_fn: Loss function of difference between soft student predictions and soft teacher predictions
      lambda: weight to student_loss_fn and 1-alpha to distillation_loss_fn
      temperature: Temperature for softening probability distributions. Larger temperature gives softer distributions.
      """
      super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
      self.student_loss_fn = student_loss_fn
      self.distillation_loss_fn = distillation_loss_fn

      #hyper-parameters
      self.Lambda = Lambda
      self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher (professor)
        teacher_predictions = self.teacher(x, training=False)

        student_predictions = self.student(x, training=True)
        tf.print("Teacher predictions shape:", tf.shape(teacher_predictions))
        tf.print("Student predictions shape:", tf.shape(student_predictions))
        tf.print("Ground truth y shape:", tf.shape(y))

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.Lambda * student_loss + (1 - self.Lambda) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        
        return results


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

elif model_name == "EfficientNetV2B0-Distilled":
    # Teacher model
    teacher_model = build_efficient_net(
        image_height,
        image_width,
        num_channels,
        num_classes,
        model_name,
        train_base=train_base,
    )
    student_model = build_student_model(image_height, image_width, num_channels, num_classes, model_name='student_distill')
    model = Distiller(teacher=teacher_model, student=student_model, model_name=model_name)
    student_loss = keras.losses.sparse_categorical_crossentropy
    distillation_loss = keras.losses.CategoricalCrossentropy(from_logits=False)
    # Optimizer
    #optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer = keras.optimizers.Adam()
    # Compile
    model.compile(
        optimizer=optimizer, 
        student_loss_fn=student_loss,
        distillation_loss_fn=distillation_loss,
        metrics=["accuracy"])

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

if model_name == "EfficientNetV2B0-Distilled":
    training_history = training_results.history
    evaluation_results = evaluate_model(model, validation_data,
               training_history, execution_time, learning_rate, batch_size, epochs, optimizer,
               save=False,
               loss_metrics=["student_loss","distillation_loss","val_student_loss"],
               acc_metrics=["accuracy","val_accuracy"])
    save_model(student_model, training_history, execution_time, learning_rate, epochs, optimizer, evaluation_results)
    models_folder = "models" # distil_models / models
    models_metrics_list = glob(models_folder+"/*_metrics.json")

    all_models_metrics = []
    for mm_file in models_metrics_list:
        with open(mm_file) as json_file:
            model_metrics = json.load(json_file)
            model_metrics["name"] = mm_file.replace(models_folder+"/","").replace("_model_metrics.json","")
            all_models_metrics.append(model_metrics)

    # Load metrics to dataframe
    view_metrics = pd.DataFrame(data=all_models_metrics)

    # Format columns
    view_metrics['accuracy'] = view_metrics['accuracy']*100
    view_metrics['accuracy'] = view_metrics['accuracy'].map('{:,.2f}%'.format)

    view_metrics['trainable_parameters'] = view_metrics['trainable_parameters'].map('{:,.0f}'.format)
    view_metrics['execution_time'] = view_metrics['execution_time'].map('{:,.2f} mins'.format)
    view_metrics['loss'] = view_metrics['loss'].map('{:,.2f}'.format)
    view_metrics['model_size'] = view_metrics['model_size']/1000000
    view_metrics['model_size'] = view_metrics['model_size'].map('{:,.3f} MB'.format)

    view_metrics = view_metrics.sort_values(by=['accuracy'],ascending=False)
    view_metrics.head()

    # wandb log
    print(f"trainable_parameters: {view_metrics['trainable_parameters']}")
    print(f"execution_time: {view_metrics['execution_time']}")
    print(f"loss: {view_metrics['loss']}")
    print(f"accuracy: {view_metrics['accuracy']}")
    print(f"model_size: {view_metrics['model_size']}")
    print(f"learning_rate: {view_metrics['learning_rate']}")
    print(f"batch_size: {view_metrics['batch_size']}")
    print(f"epochs: {view_metrics['epochs']}")
    print(f"optimizer: {view_metrics['optimizer']}")
    print(f"name: {view_metrics['name']}")

# Update W&B
wandb.config.update({"execution_time": execution_time})
# Close the W&B run
wandb.run.finish()

print("Training Job Complete")
