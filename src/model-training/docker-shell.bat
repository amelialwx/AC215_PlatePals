set -e

export IMAGE_NAME=model-training-cli
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../../secrets/
export GCS_BUCKET_URI="gs://platepals_trainer"
export GCP_PROJECT="ac215-399520"

docker build -t $IMAGE_NAME -f Dockerfile .
#docker build -t $IMAGE_NAME --platform=linux/arm64/v8 -f Dockerfile .

winpty docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/model-trainer.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_BUCKET_URI=$GCS_BUCKET_URI \
-e WANDB_KEY=$WANDB_KEY \
$IMAGE_NAME