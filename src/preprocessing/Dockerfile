FROM python:3.9-slim-buster

# Install dependencies
COPY requirements.txt /
RUN pip install -r requirements.txt

# Copy your scripts
COPY preprocess.py /

# Set environment variables for GCP
ENV GCP_PROJECT=ai5-project
ENV GCP_ZONE=us-central1-a
ENV GOOGLE_APPLICATION_CREDENTIALS=/secrets/bucket-reader.json

CMD ["python", "preprocess.py"]
