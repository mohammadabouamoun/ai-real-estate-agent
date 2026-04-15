# Use an official lightweight Python 3.11 image as base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (gcc is required for some Python packages like numpy)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies (no cache to keep image smaller)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY src/ ./src/

# Copy the trained model and imputation parameters
COPY models/ ./models/
COPY processed/ ./processed/

# Expose port 8000 (the port FastAPI will listen on)
EXPOSE 8000

# Command to start the FastAPI server inside the container
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]