# Use a multi-stage build approach to minimize vulnerabilities

# Builder stage for creating a virtual environment and installing packages
FROM python:3.12-slim-bookworm AS builder

# Upgrade system packages to address vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

COPY requirements.txt .

# Install build dependencies and Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        python3-dev \
        libpq-dev \
        build-essential \
        libssl-dev \
        libffi-dev && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade setuptools wheel && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Final stage with minimal image and pre-built dependencies
FROM python:3.12-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Runtime dependencies only - ensure all libraries needed at runtime are included
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpq5 \
        libssl3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Create a non-root user
    useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app

# Copy application code
COPY ./src /app/src

WORKDIR /app/src

USER appuser

EXPOSE 5000

# Use the CMD array syntax to ensure proper signal handling
CMD ["python", "app.py"]
