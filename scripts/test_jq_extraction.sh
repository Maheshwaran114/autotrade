#!/bin/bash
# Test script for an alternative terraform output approach

# Create a simple terraform.tfstate-like file for testing
echo "Creating test terraform.tfstate with an IP..."
cat > test.tfstate << EOF
{
  "version": 4,
  "terraform_version": "1.0.0",
  "resources": [
    {
      "mode": "managed",
      "type": "digitalocean_droplet",
      "name": "bn_trading",
      "instances": [
        {
          "attributes": {
            "ipv4_address": "64.227.149.109"
          }
        }
      ]
    }
  ]
}
EOF

echo "Testing jq extraction method:"
IP_JQ=$(jq -r '.resources[] | select(.type=="digitalocean_droplet") | .instances[0].attributes.ipv4_address' test.tfstate)
echo "Extracted IP with jq: $IP_JQ"

# Cleanup
rm test.tfstate

echo "Test complete"
