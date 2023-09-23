cd ../preprocessing
docker build -t preprocess-image .
docker run -e GCS_BUCKET_NAME=platepals_data preprocess-image
