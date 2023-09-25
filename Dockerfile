
# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container
COPY . .

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install flask

# Define environment variable for Flask to run in production mode
ENV FLASK_ENV=production

# Run the Flask API when the container launches
CMD ["python", "./notebook/api.py"]
