# Use the official Python image as a base image
FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install the necessary Python packages
RUN apt-get update && apt-get install -y ffmpeg
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files to the working directory
COPY . /app/

# Expose the port the app runs on
EXPOSE 8081

# Command to run the Flask app
CMD ["python", "flaskinference.py"]

