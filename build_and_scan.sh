#!/bin/bash
# This script builds and scans the Docker image for security vulnerabilities

set -e

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting build and security scan process...${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo -e "${RED}Error: Docker is not running.${NC}"
  exit 1
fi

# Build the Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -t bn-trading:latest .

echo -e "${GREEN}Docker image built successfully!${NC}"

# Check if Trivy is installed for vulnerability scanning
if ! command -v trivy &> /dev/null; then
    echo -e "${YELLOW}Trivy not found. Installing Trivy vulnerability scanner...${NC}"
    
    # Install Trivy based on OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install aquasecurity/trivy/trivy
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        sudo apt-get install -y wget apt-transport-https gnupg lsb-release
        wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
        echo deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main | sudo tee -a /etc/apt/sources.list.d/trivy.list
        sudo apt-get update
        sudo apt-get install -y trivy
    else
        echo -e "${RED}Unsupported OS for automatic Trivy installation. Please install manually: https://aquasecurity.github.io/trivy/v0.34/getting-started/installation/${NC}"
    fi
fi

# Scan the image for vulnerabilities
echo -e "${YELLOW}Scanning image for vulnerabilities...${NC}"
trivy image --severity HIGH,CRITICAL bn-trading:latest

echo -e "${GREEN}Build and scan process completed!${NC}"
echo -e "${YELLOW}Note: Some base image vulnerabilities may be unavoidable, but should be monitored.${NC}"
echo -e "${YELLOW}Focus on fixing HIGH and CRITICAL vulnerabilities in dependencies you control.${NC}"
echo -e "${GREEN}To run the application locally:${NC}"
echo -e "  docker-compose up -d"
