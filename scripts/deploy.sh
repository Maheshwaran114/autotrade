#!/bin/bash

# Bank Nifty Trading System Deployment Script
# This script deploys the application to the specified environment

set -e

# Check if environment is provided
if [ -z "$1" ]; then
  echo "Error: Environment not specified"
  echo "Usage: $0 <environment> [version]"
  echo "  environment: dev, preprod, prod"
  echo "  version: optional. defaults to latest"
  exit 1
fi

ENV="$1"
VERSION="${2:-latest}"

echo "Deploying Bank Nifty Trading System to $ENV environment (version: $ENV-$VERSION)"

# Set variables based on environment
case "$ENV" in
  dev)
    TFVARS="dev.tfvars"
    ;;
  preprod)
    TFVARS="preprod.tfvars"
    ;;
  prod)
    TFVARS="prod.tfvars"
    ;;
  *)
    echo "Error: Unknown environment '$ENV'"
    echo "Valid environments: dev, preprod, prod"
    exit 1
    ;;
esac

# Navigate to terraform directory
cd "$(dirname "$0")/../infra"

# Initialize and apply Terraform
echo "Applying Terraform configuration for $ENV..."
terraform init
terraform apply -var-file="$TFVARS" -auto-approve

# Get the IP address of the droplet
DROPLET_IP=$(terraform output -raw droplet_ip)

echo "Deployment complete! Application is available at:"
echo "http://$DROPLET_IP:5000/"
