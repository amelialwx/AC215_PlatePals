#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Define some environment variables
# Automatic export to the environment of subsequently executed commands
# source: the command 'help export' run in Terminal
export IMAGE_NAME="preprocessing-preprocess-image"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../../secrets/
export GCS_BUCKET_NAME="ac215-platepals"
export GCP_PROJECT="My Project 1919"
export GCP_ZONE="us-central1-a"
export GOOGLE_APPLICATION_CREDENTIALS=/../secrets/data-service-account.json

# Check to see if path to secrets is correct
if [ ! -f "$SECRETS_DIR/data-service-account.json" ]; then
    echo "data-service-account.json not found at the path you have provided."
    exit 1
fi

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .

echo "Host GOOGLE_APPLICATION_CREDENTIALS: $GOOGLE_APPLICATION_CREDENTIALS"

# Run the container
# Run Docker with an initial command to check for the secret before proceeding
docker run --rm --name $IMAGE_NAME -i \
--mount type=bind,source="$BASE_DIR",target=/app \
--mount type=bind,source="$SECRETS_DIR",target=/secrets \
-e GOOGLE_APPLICATION_CREDENTIALS="$GOOGLE_APPLICATION_CREDENTIALS" \
-e GCP_PROJECT="$GCP_PROJECT" \
-e GCP_ZONE="$GCP_ZONE" \
-e GCS_BUCKET_NAME="$GCS_BUCKET_NAME" \
-e DEV=1 $IMAGE_NAME /bin/bash -c "if [ ! -f $GOOGLE_APPLICATION_CREDENTIALS ]; then echo 'data-service-account.json not found at the path you have provided.' && exit 1; else python /app/preprocess.py; fi"
