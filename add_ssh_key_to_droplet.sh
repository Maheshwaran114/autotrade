#!/bin/bash
# Script to add your SSH key to the DigitalOcean droplet
# You'll need the root password for this to work

# Configuration
DROPLET_IP="64.227.129.85"
PUBLIC_KEY_FILE="$HOME/.ssh/id_bn_trading.pub"

# Check if the public key file exists
if [ ! -f "$PUBLIC_KEY_FILE" ]; then
  echo "❌ Error: Public key file not found at $PUBLIC_KEY_FILE"
  exit 1
fi

# Get the public key content
PUBLIC_KEY=$(cat "$PUBLIC_KEY_FILE")

echo "===== Adding SSH Key to DigitalOcean Droplet ====="
echo "This script will add your public key to the authorized_keys file on your droplet."
echo "You'll need the root password for your droplet."
echo

# Display instructions
echo "Public key to add:"
echo "$PUBLIC_KEY"
echo

# Create command to add the key
ADD_KEY_CMD="mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '$PUBLIC_KEY' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"

echo "Connecting to $DROPLET_IP to add your key..."
echo "When prompted, enter the root password for your droplet."
echo

# Connect and add the key
ssh -o StrictHostKeyChecking=no root@$DROPLET_IP "$ADD_KEY_CMD"

# Check the result
if [ $? -eq 0 ]; then
  echo "✅ Key added successfully!"
  echo "Now try running the test_ssh_connection.sh script to verify the connection."
else
  echo "❌ Failed to add key. Please check your password and try again."
fi
