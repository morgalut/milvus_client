# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY Requirements.txt /app/

# Install the required packages
RUN pip install --no-cache-dir -r Requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Set environment variables
ENV MILVUS_URI=https://in03-9b9fce0682a5279.api.gcp-us-west1.zillizcloud.com
ENV MILVUS_TOKEN=api_key

# Command to run your application
CMD ["python", "src/main.py"]

