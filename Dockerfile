# Use an official Python image with Linux as the base
FROM python:3.11-slim-bullseye

# Install C++ libraries and other dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    ffmpeg libsm6 libxext6 cmake \
    && rm -rf /var/lib/apt/lists/*


# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Install any necessary Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Expose the port the app runs on
EXPOSE 8000

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
