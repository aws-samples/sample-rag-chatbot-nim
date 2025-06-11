# Use an official Python runtime as the base image
FROM python:3.12-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script into the container
COPY . .

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Command to run the script
CMD ["python", "gradio_app.py"]