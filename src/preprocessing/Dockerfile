# Start from an NVIDIA-aware base image.
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04 as base

# Install Python 3.9 and other dependencies.
FROM base as python
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.9 python3-pip && \
    python3.9 -m pip install --upgrade pip && \
    ln -s /usr/bin/python3.9 /usr/bin/python  # Symlink python to python3.9

# Copy requirements and install Python packages.
FROM python as dependencies
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

# Final image
FROM dependencies as final
WORKDIR /app
COPY . /app

# Your entry point
ENTRYPOINT ["/bin/bash", "./docker-entrypoint.sh"]
