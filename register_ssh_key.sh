#!/bin/bash
# Script to add your SSH key to DigitalOcean account

# Configuration
API_TOKEN="YOUR_DIGITALOCEAN_TOKEN" # Replace with your token when running
PUBLIC_KEY_FILE="$HOME/.ssh/id_bn_trading.pub"
KEY_NAME="bn-trading-key-$(date +%Y%m%d)"

# Check if the public key file exists
if [ ! -f "$PUBLIC_KEY_FILE" ]; then
  echo "❌ Error: Public key file not found at $PUBLIC_KEY_FILE"
  exit 1
fi

# Get the public key content
PUBLIC_KEY=$(cat "$PUBLIC_KEY_FILE")

echo "===== Adding SSH Key to DigitalOcean Account ====="
echo "This script will add your public key to your DigitalOcean account."
echo "Then you can use it when creating new droplets."
echo

# Display key info
echo "Public key to add:"
echo "$PUBLIC_KEY"
echo
echo "Key name in DigitalOcean: $KEY_NAME"
echo

# Add the key to DigitalOcean
echo "Adding key to DigitalOcean..."
RESPONSE=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_TOKEN" \
  -d "{\"name\":\"$KEY_NAME\",\"public_key\":\"$PUBLIC_KEY\"}" \
  "https://api.digitalocean.com/v2/account/keys")

# Check for errors
if echo "$RESPONSE" | grep -q "id"; then
  # Success
  KEY_ID=$(echo "$RESPONSE" | grep -o '"id":[0-9]*' | grep -o '[0-9]*')
  echo "✅ Key added successfully with ID: $KEY_ID"
  echo
  echo "Important: Save this SSH key ID. You'll need it for your GitHub Actions workflow."
  echo "SSH_KEY_ID=$KEY_ID"
  
  # Save to a file for reference
  echo "export SSH_KEY_ID=$KEY_ID" > ./ssh_key_id.txt
  echo "Key ID saved to ssh_key_id.txt"
  
  echo
  echo "Next steps:"
  echo "1. Update your terraform.tfvars file with this SSH key ID"
  echo "2. Add the SSH_KEY_ID secret to your GitHub repository"
else
  # Error
  echo "❌ Failed to add key to DigitalOcean:"
  echo "$RESPONSE"
  
  # Check if key already exists
  if echo "$RESPONSE" | grep -q "already in use"; then
    echo
    echo "This key appears to already be registered. Fetching existing keys..."
    curl -s -X GET \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $API_TOKEN" \
      "https://api.digitalocean.com/v2/account/keys" | \
      grep -A 5 -B 5 "$(echo "$PUBLIC_KEY" | cut -d' ' -f2)"
  fi
fi
