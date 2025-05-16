#!/bin/bash
# Script to generate and display a new SSH key pair for DigitalOcean deployment

# Text formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BOLD}SSH Key Generator for DigitalOcean Deployment${NC}\n"

KEY_NAME="do_deployment_key"
if [ -n "$1" ]; then
    KEY_NAME="$1"
fi

echo -e "Generating a new SSH key pair named: ${BLUE}${KEY_NAME}${NC}"

# Create a new key in PEM format (works better with GitHub Actions)
ssh-keygen -t rsa -b 4096 -m PEM -f "./${KEY_NAME}" -q -N ""

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to generate SSH key.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ SSH key pair generated successfully!${NC}\n"

# Get the public key
PUBLIC_KEY=$(cat "./${KEY_NAME}.pub")

# Format private key for GitHub Actions
PRIVATE_KEY=$(cat "./${KEY_NAME}")

echo -e "${BOLD}Next Steps:${NC}"
echo -e "${YELLOW}1. Add the following public key to DigitalOcean${NC}"
echo -e "   Go to: DigitalOcean Dashboard > Account > Security > SSH Keys > Add SSH Key"
echo -e "   Name it something like 'GitHub Actions Deployment Key'"
echo -e "   Paste this public key:${NC}"
echo
echo -e "${BLUE}$PUBLIC_KEY${NC}"
echo

echo -e "${YELLOW}2. Add the private key as a GitHub secret${NC}"
echo -e "   Go to: GitHub Repository > Settings > Secrets > New repository secret"
echo -e "   Name: SSH_PRIVATE_KEY"
echo -e "   Value: Copy the entire private key below including BEGIN/END lines"
echo
echo -e "${BOLD}Private key for GitHub secret:${NC}"
echo -e "${BLUE}$PRIVATE_KEY${NC}"
echo

echo -e "${YELLOW}3. Update the terraform.tfvars file${NC}"
echo -e "   After adding the key to DigitalOcean, note the SSH key ID from the UI"
echo -e "   Update the ssh_key_ids value in terraform.tfvars with the new ID"
echo

echo -e "${YELLOW}4. Keep the private key secure!${NC}"
echo -e "   You may want to backup these files securely and then remove them from this directory"
echo -e "   ${BOLD}Important:${NC} The private key should be kept confidential"

echo -e "\n${GREEN}All done! Follow the steps above to complete your setup.${NC}"
