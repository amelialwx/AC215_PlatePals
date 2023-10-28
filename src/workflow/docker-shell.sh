#!/bin/bash

# set -e

export IMAGE_NAME="platepals-workflow"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../../secrets/
export GCP_PROJECT="ac215-399520" # CHANGE THIS
export GCS_BUCKET_NAME="platepals_trainers" # CHANGE THIS
export GCS_SERVICE_ACCOUNT="data-service-account@ac215-399520.iam.gserviceaccount.com" #CHANGE THIS
export GCP_REGION="us-east1"
export GCS_PACKAGE_URI="gs://platepals_trainers" # CHANGE THIS
export GOOGLE_APPLICATION_CREDENTIALS="/../secrets/data-service-account.json"

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .
# M1/2 chip macs use this line
# docker build -t $IMAGE_NAME --platform=linux/amd -f Dockerfile .

# Run Container
winpty docker run --rm --name $IMAGE_NAME -ti \
--mount type=bind,source="$BASE_DIR",target=/app \
--mount type=bind,source="$SECRETS_DIR",target=/secrets \
--mount type=bind,source="$BASE_DIR/../preprocessing",target=/preprocessing \
-e GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
-e GCS_SERVICE_ACCOUNT=$GCS_SERVICE_ACCOUNT \
-e GCP_REGION=$GCP_REGION \
-e GCS_PACKAGE_URI=$GCS_PACKAGE_URI \
$IMAGE_NAME

