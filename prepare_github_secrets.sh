#!/bin/bash
# Script to prepare values for GitHub Secrets
# This script will help you get the values needed for GitHub Secrets

# Configuration
SSH_KEY_ID="47586025"
SSH_PRIVATE_KEY_PATH="$HOME/.ssh/id_bn_trading"
GITHUB_SECRETS_FILE="github_secrets.txt"

echo "===== GitHub Secrets Helper ====="
echo "This script will help you get the values needed for GitHub Secrets."
echo

# Check if private key exists
if [ ! -f "$SSH_PRIVATE_KEY_PATH" ]; then
  echo "❌ Error: SSH private key not found at $SSH_PRIVATE_KEY_PATH"
  exit 1
fi

# Get the private key content
PRIVATE_KEY=$(cat "$SSH_PRIVATE_KEY_PATH")

# Create a file with instructions and values
cat > "$GITHUB_SECRETS_FILE" << EOF
# GitHub Secrets for Bank Nifty Trading Workflow

Follow these steps to add the required secrets to your GitHub repository:

1. Go to your GitHub repository
2. Click on "Settings" > "Secrets and variables" > "Actions"
3. Click "New repository secret"
4. Add the following secrets:

## SSH_KEY_ID
Name: SSH_KEY_ID
Value: $SSH_KEY_ID

## SSH_PRIVATE_KEY
Name: SSH_PRIVATE_KEY
Value:
$PRIVATE_KEY

## DIGITALOCEAN_TOKEN
Name: DIGITALOCEAN_TOKEN
Value: YOUR_DO_TOKEN_HERE

## DOCKERHUB_USERNAME (if you have one)
Name: DOCKERHUB_USERNAME
Value: [Your Docker Hub username]

## DOCKERHUB_TOKEN (if you have one)
Name: DOCKERHUB_TOKEN
Value: [Your Docker Hub access token]

Remember: These values are sensitive! Do not share this file.
EOF

echo "✅ Instructions and values saved to $GITHUB_SECRETS_FILE"
echo "Follow the instructions in this file to add the secrets to your GitHub repository."
echo
echo "⚠️ IMPORTANT: Keep this file secure as it contains sensitive information."
echo "Delete it after adding the secrets to GitHub."
