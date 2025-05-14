#!/bin/bash
# This script builds and scans the Docker image for security vulnerabilities

set -e

echo "Building Docker image..."
docker build -t autotrade:latest .

echo "Installing Trivy scanner for vulnerability scanning..."
brew install aquasecurity/trivy/trivy

echo "Scanning Docker image for vulnerabilities..."
trivy image autotrade:latest

echo "Note: Some base image vulnerabilities may be unavoidable, but should be monitored."
echo "Focus on fixing HIGH and CRITICAL vulnerabilities in dependencies you control."
echo "Complete! The image has been built and scanned."
