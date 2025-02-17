# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies (for building wheels and libraries like numpy)
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the ports for API and metrics
EXPOSE 8000 9090

# Run the app using the `run.py` script
CMD ["python", "scripts/run.py", "--mode", "training"]  # Set the mode (either "training" or "real_time")
