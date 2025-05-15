# Bank Nifty Options Trading System

An automated trading system for Bank Nifty options with integrated analytics and execution capabilities.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Architecture](#architecture)
- [Development](#development)
- [Deployment](#deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

## CI/CD Pipeline

This project uses GitHub Actions for continuous integration and deployment to DigitalOcean infrastructure.

### Pipeline Overview

1. **Build & Push Docker Image**: Builds the application Docker image and pushes it to DockerHub
2. **Terraform Apply**: Provisions infrastructure on DigitalOcean using Terraform
3. **Deploy Application**: Deploys the application to the provisioned infrastructure

### Required Secrets

The following secrets need to be set in your GitHub repository:

- `DOCKERHUB_USERNAME`: Your DockerHub username
- `DOCKERHUB_TOKEN`: DockerHub access token
- `DIGITALOCEAN_TOKEN`: DigitalOcean API token
- `SSH_KEY_ID`: The ID of your SSH key in DigitalOcean
- `SSH_PRIVATE_KEY`: Your private SSH key for accessing the droplet

### Local Development

For local development and testing:

```bash
# Build and scan the Docker image for vulnerabilities
./build_and_scan.sh

# Run the application locally
docker-compose up -d
```

### Troubleshooting

For help with CI/CD pipeline issues, see the [Troubleshooting Guide](docs/TROUBLESHOOTING.md).

# Last updated: Wed May 15 12:30:00 IST 2025
