#!/bin/bash
cd ../../preprocessing
docker build -t preprocess-image .
docker run -e GOOGLE_APPLICATION_CREDENTIALS='/secrets/data-service-account.json' preprocess-image
