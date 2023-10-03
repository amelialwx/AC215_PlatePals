#!/bin/bash

set -e

export IMAGE_NAME=model-training-cli
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../../secrets/
export GCS_BUCKET_URI="gs://platepals_trainer_zqiu"
export GCS_BUCKET_DATA_URI="gs://platepals_data_zqiu"
export GCP_PROJECT="platepals-400123"
#export WANDB_KEY=28624533918ab8da6d171e5f154c47cdbb3dbec8


# Build the image based on the Dockerfile
#docker build -t $IMAGE_NAME -f Dockerfile .
# M1/2 chip macs use this line
docker build -t $IMAGE_NAME --platform=linux/arm64/v8 -f Dockerfile .

# Run Container
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/model-trainer-415.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_BUCKET_URI=$GCS_BUCKET_URI \
-e GCS_BUCKET_DATA_URI = $GCS_BUCKET_DATA_URI\
-e WANDB_KEY=$WANDB_KEY \
$IMAGE_NAME