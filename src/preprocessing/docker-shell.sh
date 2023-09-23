#!/bin/bash
# Navigate to the directory where the Dockerfile is located.
cd ../preprocessing

# Build the Docker image.
docker build -t preprocess-image .

# Run the Docker container, passing in the environment variable.
docker run -e GCS_BUCKET_NAME=platepals_data preprocess-image
