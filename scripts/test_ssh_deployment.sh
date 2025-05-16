#!/bin/bash
# Test script for setting up SSH locally and deploying to a DigitalOcean droplet
# This script simulates the GitHub Actions workflow but runs locally

set -e # Exit on error

echo "===== Droplet SSH Test Tool ====="
echo

# Check if DO_TOKEN is provided
if [ -z "$1" ]; then
  echo "Error: Missing DigitalOcean token"
  echo "Usage: $0 <do_token> <droplet_ip> [ssh_private_key_file]"
  exit 1
fi

DO_TOKEN="$1"

# Check if DROPLET_IP is provided
if [ -z "$2" ]; then
  echo "Error: Missing droplet IP address"
  echo "Usage: $0 <do_token> <droplet_ip> [ssh_private_key_file]"
  exit 1
fi

DROPLET_IP="$2"

# Set up SSH key
if [ -n "$3" ]; then
  SSH_KEY="$3"
  if [ ! -f "$SSH_KEY" ]; then
    echo "Error: SSH key file not found: $SSH_KEY"
    exit 1
  fi
  
  echo "Using provided SSH key: $SSH_KEY"
else
  # Generate a temporary SSH key
  echo "No SSH key provided. Generating a temporary key..."
  SSH_KEY="./temp_ssh_key"
  ssh-keygen -t rsa -b 4096 -f "$SSH_KEY" -N "" -q
  echo "Generated temporary SSH key: $SSH_KEY"
fi

# Extract public key
echo "Extracting public key..."
PUBKEY=$(ssh-keygen -y -f "$SSH_KEY" || { echo "Failed to extract public key"; exit 1; })
echo "$PUBKEY" > "${SSH_KEY}.pub"
echo "Public key extracted and saved to ${SSH_KEY}.pub"

# Set proper permissions
chmod 600 "$SSH_KEY"
chmod 644 "${SSH_KEY}.pub"

# Get droplet ID
echo "Getting droplet ID for $DROPLET_IP..."
DROPLET_ID=$(curl -s -X GET \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DO_TOKEN" \
  "https://api.digitalocean.com/v2/droplets?per_page=200" | \
  grep -o '"id":[0-9]*,"name":"[^"]*","memory":[0-9]*,"vcpus":[0-9]*,"disk":[0-9]*,"locked":\w*,"status":"active","kernel":null,"created_at":"[^"]*","features":\[[^]]*\],"backup_ids":\[\],"next_backup_window":null,"snapshot_ids":\[\],"image":{[^}]*},"volume_ids":\[\],"size":{[^}]*},"size_slug":"[^"]*","networks":{"v4":\[{"ip_address":"'"$DROPLET_IP"'"' | \
  grep -o '"id":[0-9]*' | head -1 | cut -d':' -f2)

if [ -z "$DROPLET_ID" ]; then
  echo "Error: Could not find a droplet with IP: $DROPLET_IP"
  exit 1
fi

echo "Found droplet ID: $DROPLET_ID"

# Add the SSH key directly via console access
echo "Adding SSH key to droplet via console access..."
CONSOLE_COMMAND="mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '$PUBKEY' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && echo 'Key added successfully' && systemctl restart sshd"

echo "Sending command to droplet console..."
CONSOLE_RESPONSE=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DO_TOKEN" \
  -d "{\"type\":\"console\", \"command\":\"$CONSOLE_COMMAND\"}" \
  "https://api.digitalocean.com/v2/droplets/$DROPLET_ID/actions")

echo "Console API response:"
echo "$CONSOLE_RESPONSE"

# Extract action ID
ACTION_ID=$(echo "$CONSOLE_RESPONSE" | grep -o '"id":[0-9]*' | head -1 | cut -d':' -f2)
if [ -n "$ACTION_ID" ]; then
  echo "Console action initiated with ID: $ACTION_ID"
else
  echo "Failed to initiate console action"
  exit 1
fi

# Wait for the key to be active
echo "Waiting 15 seconds for the key to be active..."
sleep 15

# Test SSH connection
echo "Testing SSH connection..."
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$SSH_KEY" root@$DROPLET_IP 'echo "SSH connection successful!"'

# If we get here, SSH was successful
echo "✅ SSH connection established successfully!"
echo "You can now deploy with:"
echo "ssh -i $SSH_KEY root@$DROPLET_IP"

exit 0
