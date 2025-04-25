# Use an official Python runtime as a parent image
FROM  python:3.10-slim

# Install build dependencies and required libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libleptonica-dev \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libglib2.0-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    poppler-utils \
    git \
    tesseract-ocr \
    wget \
    libmagic1 \
    libmagic-dev \
    libjpeg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make sure gpu is not used
ENV CUDA_VISIBLE_DEVICES=""

RUN pip install h5py==3.6.0

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Create a temp directory for holding the pdf files that need to be processed
RUN mkdir -p /app/tmp

# Install the necessary Python packages
RUN pip install --no-cache-dir -r parsing-requirements.txt
RUN pip install -r parsing-requirements.txt --use-deprecated=legacy-resolver
RUN pip install "numpy<2"
RUN pip install "tokenizers>=0.21,<0.22"
RUN pip install --upgrade transformers
RUN pip install python-magic
RUN pip install pi_heif
RUN pip install --upgrade --quiet "unstructured[all-docs]"

# Expose the port Flask runs on
EXPOSE 5000

# Define environment variables for Flask
ENV FLASK_APP=parsingAPI.py
ENV FLASK_ENV=development

# Run the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]