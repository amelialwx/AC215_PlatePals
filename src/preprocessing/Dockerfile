# Use the official Debian-hosted Python image
FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Tell pipenv where the shell is. 
ENV PYENV_SHELL=/bin/bash

# Ensure we have an up-to-date baseline, install dependencies
RUN set -ex && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends build-essential git && \
    pip install --no-cache-dir --upgrade pip

# Copy requirements and install Python packages
COPY requirements.txt /
RUN pip install -r /requirements.txt

# Add everything
ADD . /app

# Entry point
ENTRYPOINT ["/bin/bash", "./docker-entrypoint.sh"]