#!/bin/bash
# Script to list SSH keys in DigitalOcean account

# Configuration
API_TOKEN="YOUR_DIGITALOCEAN_TOKEN" # Replace with your token when running

echo "===== Listing SSH Keys in DigitalOcean Account ====="

# List all keys
echo "Fetching keys from DigitalOcean..."
RESPONSE=$(curl -s -X GET \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_TOKEN" \
  "https://api.digitalocean.com/v2/account/keys")

# Print raw response for debugging
echo "Raw response:"
echo "$RESPONSE"
echo

# Check for errors
if echo "$RESPONSE" | grep -q "ssh_keys"; then
  # Success
  echo "SSH Keys found:"
  echo "$RESPONSE" | grep -o '"id":[0-9]*' | grep -o '[0-9]*'
  echo "$RESPONSE" | grep -o '"name":"[^"]*"' | grep -o ':"[^"]*"' | tr -d :'\"'
  echo "$RESPONSE" | grep -o '"fingerprint":"[^"]*"' | grep -o ':"[^"]*"' | tr -d :'\"'
  
  # Look for specific fingerprint
  if echo "$RESPONSE" | grep -q "f4:1a:7a:a9:a6:d7:99:cc:f9:68:82:07:bb:d7:d7:80"; then
    echo
    echo "Found key with fingerprint f4:1a:7a:a9:a6:d7:99:cc:f9:68:82:07:bb:d7:d7:80"
    KEY_ID=$(echo "$RESPONSE" | grep -B 5 "f4:1a:7a:a9:a6:d7:99:cc:f9:68:82:07:bb:d7:d7:80" | grep -o '"id":[0-9]*' | head -1 | grep -o '[0-9]*')
    echo "SSH_KEY_ID=$KEY_ID"
    echo "export SSH_KEY_ID=$KEY_ID" > ./ssh_key_id.txt
    echo "Key ID saved to ssh_key_id.txt"
  fi
else
  # Error
  echo "❌ Failed to list keys from DigitalOcean:"
  echo "$RESPONSE"
fi
