FROM python:3.9-slim-buster

# Set working directory
WORKDIR /app

# Install necessary packages and libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Run the preprocessing script
ENTRYPOINT ["./docker-entrypoint.sh"]

