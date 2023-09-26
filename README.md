AC215-Template (Final Milestone)
==============================

AC215 - Milestone2

Project Organization
------------
      ├── LICENSE
      ├── README.md
      ├── notebooks
      ├── references
      ├── requirements.txt
      ├── setup.py
      ├── reports
      └── src
            |── preprocessing
                ├── Dockerfile
                ├── docker-entrypoint.sh
                ├── docker-shell.bat
                ├── docker-shell.sh
                ├── preprocess.py
                └── requirements.txt

--------
# AC215 - Milestone2 - PlatePals

**Team Members**
Amelia Li, Rebecca Qiu, Peter Wu

**Group Name**
PlatePals

**Project**
The goal of this project is to develop a machine learning application that accurately identifies the types of food present in a user-uploaded image. Based on the foods identified, the application will provide the user with relevant nutritional information and personalized dietary recommendations. This project will involve key phases of data preprocessing, model development, and application interface development, leveraging TensorFlow's Food-101 dataset.

### Milestone2 ###

We will primarily be using the TensorFlow food101 dataset, which contains 101,000 labeled food images covering 101 food classes. We will be mapping the macronutrients of each food class with the Kaggle Nutrition datasets and Nutrional Facts for most common foods, containing around 9000 nutritional information combined. We parked our dataset in a private Google Cloud Bucket. 

**Preprocess container**
- This container reads 4.65GB of data and resizes the image sizes and stores it back to GCP
- This container also downloads and stores nutritions data as a CSV file back to GCP
- Input to this container is source and destincation GCS location, parameters for resizing, secrets needed - via docker
- Output from this container stored at GCS location

(1) `src/preprocessing/preprocess.py`  - Here we do preprocessing on our dataset of 4.65GB, we reduce the image sizes (a parameter that can be changed later) to 128x128 for faster iteration with our process. Now we save this dataset on GCS. 

(2) `src/preprocessing/requirements.txt` - We used following packages to help us preprocess here.

(3) `src/preprocessing/Dockerfile` - This dockerfile starts with  `python:3.9-slim-buster`. This <statement> attaches volume to the docker container and also uses secrets (not to be stored on GitHub) to connect to GCS.

To run Dockerfile -
- Put `secrets` folder containing `data-service-account.json` two directories back from the `preprocessing` directory.
- Navigate to the `preprocessing` directory.
- Run `sh docker-shell.sh`