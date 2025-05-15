#!/bin/bash
# Cleanup Script for DigitalOcean Droplets
# This script will help destroy unwanted droplets

# Use provided token or load from test_config.sh if available
if [ -z "$DIGITALOCEAN_TOKEN" ] && [ -f "./test_config.sh" ]; then
  echo "Loading configuration from test_config.sh..."
  source ./test_config.sh
fi

if [ -z "$DIGITALOCEAN_TOKEN" ]; then
  echo "DIGITALOCEAN_TOKEN environment variable is not set."
  echo "Please set it with: export DIGITALOCEAN_TOKEN=your_token"
  exit 1
fi

# List all droplets
echo "=== Current Droplets ==="
DROPLETS=$(curl -s -X GET \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DIGITALOCEAN_TOKEN" \
  "https://api.digitalocean.com/v2/droplets?per_page=200")

echo "$DROPLETS" | jq -r '.droplets[] | "\(.id) - \(.name) - Created: \(.created_at) - IP: \(.networks.v4[] | select(.type=="public") | .ip_address)"'

echo
echo "Enter the ID of the droplet you want to KEEP (will delete all others with 'bn-trading-server' name):"
read -r KEEP_ID

# Confirm before deletion
echo
echo "WARNING: This will DELETE ALL OTHER bn-trading-server droplets except ID $KEEP_ID"
echo "Are you sure you want to continue? (yes/no)"
read -r CONFIRM

if [ "$CONFIRM" != "yes" ]; then
  echo "Operation cancelled."
  exit 0
fi

# Delete droplets
echo
echo "Deleting unwanted droplets..."
echo "$DROPLETS" | jq -r '.droplets[] | select(.name=="bn-trading-server" and .id != '"$KEEP_ID"') | .id' | while read -r ID; do
  echo "Deleting droplet ID: $ID"
  curl -X DELETE \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $DIGITALOCEAN_TOKEN" \
    "https://api.digitalocean.com/v2/droplets/$ID"
  echo " - Done"
done

echo
echo "Cleanup complete! Remaining droplets:"
curl -s -X GET \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DIGITALOCEAN_TOKEN" \
  "https://api.digitalocean.com/v2/droplets?per_page=200" | jq -r '.droplets[] | "\(.id) - \(.name) - IP: \(.networks.v4[] | select(.type=="public") | .ip_address)"'
