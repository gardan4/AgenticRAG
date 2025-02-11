# For creating a docker container with parsing services
FROM python:3.10-slim

# Install build dependencies and required libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    poppler-utils \
    git \
    tesseract-ocr \
    wget \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make sure GPU is not used (works more universally)
ENV CUDA_VISIBLE_DEVICES=""

RUN pip install h5py==3.6.0

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Create a temp directory for holding the pdf files that need to be processed
RUN mkdir -p /app/tmp

# Install the necessary Python packages
RUN pip install unstructured
RUN pip install unstructured-ingest[remote]
RUN pip install unstructured[pptx]
RUN pip install unstructured[markdown]
RUN pip install unstructured[pdf]
RUN pip install unstructured[docx]
RUN pip install unstructured[txt]
RUN pip install pytesseract
RUN pip install flask
RUN pip install opencv-python-headless  # Ensure OpenCV works in headless mode

# Expose the port Flask runs on
EXPOSE 5000

# Define environment variables for Flask
ENV FLASK_APP=preprocessing_api.py
ENV FLASK_ENV=development

# Run the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]
