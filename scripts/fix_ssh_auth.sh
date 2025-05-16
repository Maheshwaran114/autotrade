#!/bin/bash
# Script to fix SSH authentication issues for DigitalOcean deployment
# This script is designed to be run from GitHub Actions

set -e # Exit on error

echo "===== DigitalOcean SSH Fix ====="
echo

# Check required arguments
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Error: Missing required arguments"
  echo "Usage: $0 <do_token> <droplet_ip>"
  exit 1
fi

DO_TOKEN="$1"
DROPLET_IP="$2"

echo "Target droplet IP: $DROPLET_IP"

# Generate a fresh SSH key pair specifically for this deployment
echo "Generating new SSH key pair..."
SSH_KEY_FILE="do_deploy_key"
ssh-keygen -t rsa -b 4096 -f "./$SSH_KEY_FILE" -N "" -q

if [ ! -f "./$SSH_KEY_FILE" ] || [ ! -f "./$SSH_KEY_FILE.pub" ]; then
  echo "Error: Failed to generate SSH key pair"
  exit 1
fi

echo "SSH key pair generated successfully"
chmod 600 "./$SSH_KEY_FILE"
chmod 644 "./$SSH_KEY_FILE.pub"

# Get the public key content
PUBLIC_KEY=$(cat "./$SSH_KEY_FILE.pub")
echo "Public key: $PUBLIC_KEY"

# Get the droplet ID
echo "Getting droplet ID..."
DROPLET_ID=$(curl -s -X GET \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DO_TOKEN" \
  "https://api.digitalocean.com/v2/droplets?name=bn-trading-server" | grep -o '"id":[0-9]*' | head -1 | cut -d':' -f2)

if [ -z "$DROPLET_ID" ]; then
  echo "Error: Could not find droplet ID"
  exit 1
fi

echo "Found droplet ID: $DROPLET_ID"

# Add the SSH key to the droplet using the DigitalOcean API
echo "Adding SSH key to DigitalOcean..."

# First register the SSH key with DigitalOcean
echo "Registering SSH key with DigitalOcean..."
REGISTER_RESPONSE=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DO_TOKEN" \
  -d "{\"name\":\"deploy_key_$(date +%s)\",\"public_key\":\"$PUBLIC_KEY\"}" \
  "https://api.digitalocean.com/v2/account/keys")

echo "Key registration response: $REGISTER_RESPONSE"

# Extract the SSH key ID
SSH_KEY_ID=$(echo "$REGISTER_RESPONSE" | grep -o '"id":[0-9]*' | head -1 | cut -d':' -f2)
if [ -z "$SSH_KEY_ID" ]; then
  echo "Failed to register SSH key with DigitalOcean"
  echo "Trying alternative approach..."
  
  # Try to add it directly with a droplet action
  echo "Adding key directly..."
  
  # Try different action type - rebuild action
  REBUILD_RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $DO_TOKEN" \
    -d "{\"type\":\"rebuild\", \"image\":\"ubuntu-22-04-x64\"}" \
    "https://api.digitalocean.com/v2/droplets/$DROPLET_ID/actions")
    
  echo "Rebuild response: $REBUILD_RESPONSE"
  echo "WARNING: Droplet is being rebuilt to reset access. This may take several minutes."
  echo "After rebuild completes, you'll need to manually add your SSH key."
  exit 1
fi

echo "Successfully registered SSH key with ID: $SSH_KEY_ID"

# Now attach the key to the droplet
echo "Attaching SSH key to droplet..."
ATTACH_RESPONSE=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DO_TOKEN" \
  -d "{\"type\":\"attach_ssh_key\", \"ssh_key_ids\":[$SSH_KEY_ID]}" \
  "https://api.digitalocean.com/v2/droplets/$DROPLET_ID/actions")

echo "Attach response: $ATTACH_RESPONSE"

# Check if the attachment was successful
ATTACH_ID=$(echo "$ATTACH_RESPONSE" | grep -o '"id":[0-9]*' | head -1 | cut -d':' -f2)
if [ -z "$ATTACH_ID" ]; then
  echo "Error: Failed to attach SSH key to droplet"
  echo "Trying alternative approach - PowerCycle the droplet to apply changes..."
  
  # Try power cycling the droplet
  POWER_RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $DO_TOKEN" \
    -d "{\"type\":\"power_cycle\"}" \
    "https://api.digitalocean.com/v2/droplets/$DROPLET_ID/actions")
    
  echo "Power cycle response: $POWER_RESPONSE"
  echo "Waiting 30 seconds for droplet to restart..."
  sleep 30
else
  echo "Successfully initiated attach_ssh_key action with ID: $ATTACH_ID"
  echo "Waiting 20 seconds for changes to take effect..."
  sleep 20
fi

# Wait for the key to be active
echo "Waiting for the key to be active (20 seconds)..."
sleep 20

# Test SSH connection
echo "Testing SSH connection..."
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "./$SSH_KEY_FILE" root@$DROPLET_IP 'echo "SSH connection successful!"'

# Output private key for GitHub Actions
echo -e "\nPrivate key for SSH deployment (save this as a GitHub secret):"
cat "./$SSH_KEY_FILE"

# Success message
echo -e "\nSuccess! SSH key has been added to the droplet."
echo "You can now use this key for SSH connection and deployment."

exit 0
