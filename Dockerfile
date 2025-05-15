# Use a pre-built image with scientific Python packages to avoid long build times
FROM python:3.9-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /app

# Install system dependencies required for Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        python3-dev \
        libpq-dev \
        build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements file first for better caching
COPY requirements.txt .

# Install Python dependencies
# Use a version of pandas that has pre-built wheels for Python 3.9
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade setuptools wheel && \
    # Install pandas separately first with a compatible version
    pip install --no-cache-dir pandas==1.5.3 && \
    # Install the rest of the requirements
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./src /app/src

# Create a non-root user for running the application
RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 5000

# Use the CMD array syntax for proper signal handling
CMD ["python", "src/app.py"]
