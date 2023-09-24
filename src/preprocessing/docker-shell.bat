export IMAGE_NAME="preprocessing-preprocess-image"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../../secrets/
export GCS_BUCKET_NAME="platepals_data"
export GCP_PROJECT="AC215"
export GCP_ZONE="us-central1-a"
export GOOGLE_APPLICATION_CREDENTIALS=/secrets/data-service-account.json

if [ ! -f "$SECRETS_DIR/data-service-account.json" ]; then
    echo "data-service-account.json not found at the path you have provided."
    exit 1
fi

docker build -t $IMAGE_NAME -f Dockerfile .

echo "Host GOOGLE_APPLICATION_CREDENTIALS: $GOOGLE_APPLICATION_CREDENTIALS"

docker run --rm --name $IMAGE_NAME -i \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-e GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCP_ZONE=$GCP_ZONE \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
-e DEV=1 $IMAGE_NAME /bin/bash -c "if [ ! -f $GOOGLE_APPLICATION_CREDENTIALS ]; then echo 'data-service-account.json not found at the path you have provided.' && exit 1; else python /app/preprocess.py; fi"
