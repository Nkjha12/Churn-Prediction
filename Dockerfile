# Use a base image with Python
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy requirements file to the working directory
COPY requirements.txt .

# Install dependencies with no cache to ensure fresh installs
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir numpy scikit-learn

# Copy the rest of the app files
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Start the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
