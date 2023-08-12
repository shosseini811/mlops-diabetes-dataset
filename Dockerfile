# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY app.py /app
COPY diabetes_classifier.pkl /app
COPY requirements.txt /app

# Install the required packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Make port 7860 available to the world outside this container (Gradio default port)
EXPOSE 7860

# Run app.py when the container launches
CMD ["python", "app.py"]
