#!/bin/bash
# Script to add SSH key to an existing DigitalOcean Droplet
# This script will update an existing droplet to use your SSH key

# Configuration
API_TOKEN="YOUR_DIGITALOCEAN_TOKEN" # Replace with your token when running
DROPLET_IP="64.227.129.85"
SSH_KEY_ID="47586025"

echo "===== Update DigitalOcean Droplet with SSH Key ====="
echo "This script will update your droplet to use your SSH key."
echo

# Get droplet ID from IP
echo "Finding droplet ID for IP $DROPLET_IP..."
RESPONSE=$(curl -s -X GET \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_TOKEN" \
  "https://api.digitalocean.com/v2/droplets?per_page=200")

DROPLET_ID=$(echo "$RESPONSE" | grep -B 5 "$DROPLET_IP" | grep -o '"id":[0-9]*' | head -1 | grep -o '[0-9]*')

if [ -z "$DROPLET_ID" ]; then
  echo "❌ Error: Could not find droplet with IP $DROPLET_IP"
  echo "Raw response:"
  echo "$RESPONSE"
  exit 1
fi

echo "Found droplet ID: $DROPLET_ID"

# Add SSH key to droplet
echo "Adding SSH key ID $SSH_KEY_ID to droplet..."
RESPONSE=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_TOKEN" \
  -d "{\"type\":\"ssh_keys\",\"ssh_keys\":[$SSH_KEY_ID]}" \
  "https://api.digitalocean.com/v2/droplets/$DROPLET_ID/actions")

if echo "$RESPONSE" | grep -q "id"; then
  echo "✅ SSH key added successfully to droplet!"
  echo "You should now be able to SSH into the droplet using your key."
else
  echo "❌ Error adding SSH key to droplet:"
  echo "$RESPONSE"
fi
