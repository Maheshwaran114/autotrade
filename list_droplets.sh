#!/bin/bash
# List DigitalOcean Droplets
# This script lists all droplets in your DigitalOcean account

if [ -z "$DIGITALOCEAN_TOKEN" ]; then
  echo "DIGITALOCEAN_TOKEN environment variable is not set."
  echo "Please set it with: export DIGITALOCEAN_TOKEN=your_token"
  exit 1
fi

echo "Fetching droplets from DigitalOcean..."
curl -s -X GET \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DIGITALOCEAN_TOKEN" \
  "https://api.digitalocean.com/v2/droplets?per_page=200" | jq -r '.droplets[] | "\(.id) - \(.name) - Status: \(.status) - IP: \(.networks.v4[] | select(.type=="public") | .ip_address)"'

echo
echo "To clean up droplets, you can destroy them with:"
echo "curl -X DELETE -H \"Authorization: Bearer \$DIGITALOCEAN_TOKEN\" \"https://api.digitalocean.com/v2/droplets/DROPLET_ID\""
